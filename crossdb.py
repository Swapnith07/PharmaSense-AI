from qdrant_client import QdrantClient
from neo4j import GraphDatabase
import numpy as np
from typing import List, Dict, Optional
import logging


class SimpleDatabaseInterface:
    """
    Simple interface to perform basic operations on Vector DB and Graph DB
    - Vector DB: Find similar entities
    - Graph DB: Extract relationships
    """

    def __init__(self, config: Dict):
        """Initialize database connections"""

        self.config = config
        self.logger = self._setup_logging()

        # Connect to databases
        self._connect_qdrant()
        self._connect_neo4j()

        print("✅ SimpleDatabaseInterface initialized successfully!")

    def _setup_logging(self):
        """Setup basic logging"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def _connect_qdrant(self):
        """Connect to Qdrant vector database"""
        try:
            self.qdrant_client = QdrantClient(
                host=self.config.get('qdrant_host', 'localhost'),
                port=self.config.get('qdrant_port', 6333)
            )

            # Test connection
            collection_info = self.qdrant_client.get_collection(
                self.config.get('collection_name', 'drug_embeddings_biobert')
            )
            vector_count = collection_info.points_count

            print(f"🔗 Qdrant connected: {vector_count:,} vectors available")

        except Exception as e:
            self.logger.error(f"Qdrant connection failed: {e}")
            raise

    def _connect_neo4j(self):
        """Connect to Neo4j graph database"""
        try:
            self.neo4j_driver = GraphDatabase.driver(
                self.config.get('neo4j_uri', 'bolt://localhost:7687'),
                auth=(
                    self.config.get('neo4j_user', 'neo4j'),
                    self.config.get('neo4j_password', 'password')
                )
            )

            # Test connection
            with self.neo4j_driver.session() as session:
                result = session.run(
                    "MATCH (n) RETURN count(n) as total_nodes")
                node_count = result.single()["total_nodes"]

            print(f"🔗 Neo4j connected: {node_count:,} nodes available")

        except Exception as e:
            self.logger.error(f"Neo4j connection failed: {e}")
            raise

    def find_similar_entities(self,
                              query_vector: Optional[List[float]] = None,
                              query_text: Optional[str] = None,
                              entity_name: Optional[str] = None,
                              limit: int = 10,
                              score_threshold: float = 0.0) -> Dict:
        """
        Find similar entities in vector database

        Args:
            query_vector: Direct vector to search with
            query_text: Text to find similar entities (if you have embedding model)
            entity_name: Name of entity to find similar to (will lookup its vector first)
            limit: Maximum number of results
            score_threshold: Minimum similarity score

        Returns:
            List of similar entities with their similarity scores
        """

        try:
            collection_name = self.config.get(
                'collection_name', 'drug_embeddings_biobert')

            # Determine the query vector to use
            search_vector = None
            search_info = ""

            if query_vector is not None:
                search_vector = query_vector
                search_info = "custom vector"

            elif entity_name is not None:
                # Find the entity first to get its vector
                entity_vector = self._get_entity_vector(entity_name)
                if entity_vector:
                    search_vector = entity_vector
                    search_info = f"entity '{entity_name}'"
                else:
                    return {
                        'success': False,
                        'error': f"Entity '{entity_name}' not found in vector database",
                        'results': []
                    }

            elif query_text is not None:
                return {
                    'success': False,
                    'error': "Text search requires embedding model (not implemented in simple version)",
                    'results': []
                }

            else:
                return {
                    'success': False,
                    'error': "Must provide either query_vector, entity_name, or query_text",
                    'results': []
                }

            # Perform similarity search
            search_results = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=search_vector,
                limit=limit,
                score_threshold=score_threshold
            )

            # Format results
            similar_entities = []
            for result in search_results:
                payload = result.payload if result.payload is not None else {}
                entity_info = {
                    'entity_name': payload.get('drug_name', 'Unknown'),
                    'entity_id': payload.get('drug_id', result.id),
                    'similarity_score': float(result.score),
                    'payload': payload
                }
                similar_entities.append(entity_info)

            print(
                f"🔍 Found {len(similar_entities)} similar entities to {search_info}")

            return {
                'success': True,
                'query_info': search_info,
                'results_count': len(similar_entities),
                'results': similar_entities
            }

        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'results': []
            }

    def _get_entity_vector(self, entity_name: str) -> Optional[List[float]]:
        """Get vector for a specific entity by name"""
        try:
            collection_name = self.config.get(
                'collection_name', 'drug_embeddings_biobert')
            scroll_result = self.qdrant_client.scroll(
                collection_name=collection_name,
                limit=1000,  # Get a reasonable batch
                with_payload=True,
                with_vectors=True
            )
            entity_name_lower = entity_name.lower().strip()
            for point in scroll_result[0]:
                payload = point.payload if point.payload is not None else {}
                stored_name = payload.get('drug_name', '').lower().strip()
                if stored_name == entity_name_lower:
                    if self._is_flat_float_list(point.vector):
                        return [float(x) for x in point.vector]
            for point in scroll_result[0]:
                payload = point.payload if point.payload is not None else {}
                stored_name = payload.get('drug_name', '').lower().strip()
                if entity_name_lower in stored_name or stored_name in entity_name_lower:
                    print(f"🔍 Partial match found: '{payload.get('drug_name')}' for '{entity_name}'")
                    if self._is_flat_float_list(point.vector):
                        return [float(x) for x in point.vector]
            return None
        except Exception as e:
            self.logger.error(
                f"Failed to get vector for entity '{entity_name}': {e}")
            return None

    def _is_flat_float_list(self, vector) -> bool:
        return isinstance(vector, list) and all(isinstance(x, (float, int)) for x in vector)

    def extract_relationships(self,
                              entity_name: Optional[str] = None,
                              entity_names: Optional[List[str]] = None,
                              relationship_type: Optional[str] = None,
                              limit: int = 20) -> Dict:
        """
        Extract relationships from graph database

        Args:
            entity_name: Single entity to find relationships for
            entity_names: Multiple entities to find relationships between/among
            relationship_type: Specific relationship type to filter (e.g., 'INTERACTS_WITH')
            limit: Maximum number of relationships to return

        Returns:
            Dictionary containing relationships and metadata
        """

        try:
            with self.neo4j_driver.session() as session:
                rel_type = relationship_type if relationship_type is not None else ''
                if entity_name:
                    return self._get_single_entity_relationships(session, entity_name, rel_type, limit)
                elif entity_names and len(entity_names) >= 2:
                    return self._get_multi_entity_relationships(session, entity_names, rel_type, limit)
                else:
                    return {
                        'success': False,
                        'error': "Must provide either entity_name or entity_names (min 2)",
                        'relationships': []
                    }

        except Exception as e:
            self.logger.error(f"Graph relationship extraction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'relationships': []
            }

    def _get_single_entity_relationships(self, session, entity_name: str, relationship_type: str, limit: int) -> Dict:
        """Get relationships for a single entity including reaction nodes - FIXED VERSION"""

        # First, let's find the exact entity to ensure it exists
        entity_search_query = """
        MATCH (e:Drug)
        WHERE toLower(e.name) = toLower($entity_name) 
           OR toLower(e.id) = toLower($entity_name)
           OR toLower(e.name) CONTAINS toLower($entity_name)
        RETURN e.name as name, e.id as id
        ORDER BY 
            CASE 
                WHEN toLower(e.name) = toLower($entity_name) THEN 1
                WHEN toLower(e.id) = toLower($entity_name) THEN 2
                ELSE 3
            END
        LIMIT 1
        """
        
        # Find the exact entity first
        entity_result = session.run(entity_search_query, entity_name=entity_name)
        entity_record = entity_result.single()
        
        if not entity_record:
            return {
                'success': False,
                'error': f"Entity '{entity_name}' not found in graph database",
                'relationships': []
            }
        
        found_entity_name = entity_record['name']
        found_entity_id = entity_record['id']
        
        print(f"🔍 Found entity: '{found_entity_name}' (ID: {found_entity_id})")

        # Now search for relationships using the exact found entity
        if relationship_type and relationship_type != '':
            # Bidirectional search with specific relationship type
            cypher_query = f"""
            MATCH (target:Drug)
            WHERE toLower(target.name) = toLower($found_entity_name)
            MATCH (target)-[r:{relationship_type}]-(other:Drug)
            WHERE target <> other
            OPTIONAL MATCH (target)-[:HAS_REACTION]->(reaction:Reaction)<-[:HAS_REACTION]-(other)
            RETURN target.name as entity1_name, target.id as entity1_id,
                   type(r) as relationship_type,
                   r as relationship_props,
                   other.name as entity2_name, other.id as entity2_id,
                   reaction.id as reaction_id,
                   reaction.normalized_desc as reaction_description,
                   reaction.example as reaction_example
            LIMIT $limit
            """
        else:
            # Bidirectional search for all relationship types
            cypher_query = """
            MATCH (target:Drug)
            WHERE toLower(target.name) = toLower($found_entity_name)
            MATCH (target)-[r]-(other:Drug)
            WHERE target <> other
            OPTIONAL MATCH (target)-[:HAS_REACTION]->(reaction:Reaction)<-[:HAS_REACTION]-(other)
            RETURN target.name as entity1_name, target.id as entity1_id,
                   type(r) as relationship_type,
                   r as relationship_props,
                   other.name as entity2_name, other.id as entity2_id,
                   reaction.id as reaction_id,
                   reaction.normalized_desc as reaction_description,
                   reaction.example as reaction_example
            LIMIT $limit
            """

        result = session.run(cypher_query, found_entity_name=found_entity_name, limit=limit)

        relationships = []
        
        for record in result:
            # Extract relationship properties safely
            rel_props = {}
            if record['relationship_props']:
                try:
                    rel_props = dict(record['relationship_props'])
                except Exception as e:
                    print(f"⚠️ Error extracting relationship properties: {e}")
                    rel_props = {}

            # Extract reaction information safely
            reaction_info = None
            if record['reaction_id']:
                reaction_info = {
                    'reaction_id': record['reaction_id'],
                    'normalized_description': record['reaction_description'],
                    'example_description': record['reaction_example']
                }

            relationship_info = {
                'entity1': {
                    'name': record['entity1_name'],
                    'id': record['entity1_id']
                },
                'entity2': {
                    'name': record['entity2_name'],
                    'id': record['entity2_id']
                },
                'relationship_type': record['relationship_type'],
                'relationship_properties': rel_props,
                'reaction': reaction_info
            }
            relationships.append(relationship_info)

        print(f"🔍 Found {len(relationships)} relationships for '{found_entity_name}'")

        return {
            'success': True,
            'query_entity': entity_name,
            'found_entity': found_entity_name,
            'relationship_type_filter': relationship_type,
            'relationships_count': len(relationships),
            'relationships': relationships
        }

    def _get_multi_entity_relationships(self, session, entity_names: List[str], relationship_type: str, limit: int) -> Dict:
        """Get unique relationships between multiple entities"""
        if relationship_type and relationship_type != '':
            cypher_query = f"""
            MATCH (e1)-[r:{relationship_type}]-(e2)
            WHERE (toLower(e1.name) IN [name IN $entity_names | toLower(name)])
              AND (toLower(e2.name) IN [name IN $entity_names | toLower(name)])
              AND e1.name < e2.name  // This prevents duplicates
            OPTIONAL MATCH (e1)-[:HAS_REACTION]->(reaction:Reaction)<-[:HAS_REACTION]-(e2)
            RETURN DISTINCT e1.name as entity1_name, e1.id as entity1_id,
                   type(r) as relationship_type,
                   r.description as interaction_description,
                   e2.name as entity2_name, e2.id as entity2_id,
                   reaction.normalized_desc as reaction_description
            LIMIT $limit
            """
        else:
            cypher_query = """
            MATCH (e1)-[r]-(e2)
            WHERE (toLower(e1.name) IN [name IN $entity_names | toLower(name)])
              AND (toLower(e2.name) IN [name IN $entity_names | toLower(name)])
              AND e1.name < e2.name  // This prevents duplicates
            RETURN DISTINCT e1.name as entity1_name, e1.id as entity1_id,
                   type(r) as relationship_type,
                   r.description as interaction_description,
                   e2.name as entity2_name, e2.id as entity2_id
            LIMIT $limit
            """

        result = session.run(
            cypher_query, entity_names=entity_names, limit=limit)

        relationships = []
        seen_pairs = set()

        for record in result:
            # Create unique identifier for drug pair
            pair_key = tuple(
                sorted([record['entity1_name'], record['entity2_name']]))

            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)

                relationship_info = {
                    'entity1': {'name': record['entity1_name'], 'id': record['entity1_id']},
                    'entity2': {'name': record['entity2_name'], 'id': record['entity2_id']},
                    'relationship_type': record['relationship_type'],
                    'interaction_description': record.get('interaction_description', 'No description'),
                    'reaction_description': record.get('reaction_description', None)
                }
                relationships.append(relationship_info)

        return {
            'success': True,
            'query_entities': entity_names,
            'relationships_count': len(relationships),
            'relationships': relationships
        }

    def close_connections(self):
        """Close database connections"""
        try:
            if hasattr(self, 'neo4j_driver'):
                self.neo4j_driver.close()
            print("🔌 Database connections closed")
        except Exception as e:
            self.logger.error(f"Error closing connections: {e}")

    def debug_single_entity(self, entity_name: str) -> Dict:
        """Debug method to check entity existence and relationships"""
        try:
            with self.neo4j_driver.session() as session:
                
                # 1. Check if entity exists at all
                check_query = """
                MATCH (e:Drug)
                WHERE toLower(e.name) CONTAINS toLower($entity_name)
                   OR toLower(e.id) CONTAINS toLower($entity_name)
                RETURN e.name as name, e.id as id, labels(e) as labels
                LIMIT 10
                """
                
                result = session.run(check_query, entity_name=entity_name)
                entities = []
                for record in result:
                    entities.append({
                        'name': record['name'],
                        'id': record['id'],
                        'labels': record['labels']
                    })
                
                if not entities:
                    return {
                        'entity_exists': False,
                        'message': f"No entities found matching '{entity_name}'"
                    }
                
                # 2. Check relationships for exact match
                exact_entity = entities[0]  # Take the first match
                
                rel_check_query = """
                MATCH (e:Drug)-[r]-(other)
                WHERE toLower(e.name) = toLower($exact_name)
                RETURN type(r) as rel_type, count(*) as count
                """
                
                rel_result = session.run(rel_check_query, exact_name=exact_entity['name'])
                relationship_summary = []
                for record in rel_result:
                    relationship_summary.append({
                        'type': record['rel_type'],
                        'count': record['count']
                    })
                
                return {
                    'entity_exists': True,
                    'found_entities': entities,
                    'exact_match': exact_entity,
                    'relationship_summary': relationship_summary,
                    'total_relationships': sum(r['count'] for r in relationship_summary)
                }
                
        except Exception as e:
            return {
                'error': str(e),
                'entity_exists': False
            }


# # Example usage
# if __name__ == "__main__":

#     # Configuration
#     config = {
#         'qdrant_host': 'localhost',
#         'qdrant_port': 6333,
#         'collection_name': 'drug_embeddings_biobert',
#         'neo4j_uri': 'bolt://localhost:7687',
#         'neo4j_user': 'neo4j',
#         'neo4j_password': 'your_password'
#     }

#     try:
#         # Initialize the interface
#         db_interface = SimpleDatabaseInterface(config)

#         print("🧪 TESTING DATABASE METHODS")
#         print("="*50)

#         # Test 1: Find similar entities
#         print("1. Finding similar drugs to 'aspirin':")
#         similar_result = db_interface.find_similar_entities(
#             entity_name="aspirin", limit=5)
#         if similar_result['success']:
#             for drug in similar_result['results']:
#                 print(
#                     f"   • {drug['entity_name']} (Score: {drug['similarity_score']:.3f})")
#         else:
#             print(f"   ❌ Error: {similar_result['error']}")
#         print()

#         # Test 2: Single entity relationships
#         print("2. Finding relationships for 'metformin':")
#         single_result = db_interface.extract_relationships(
#             entity_name="metformin", limit=5)
#         if single_result['success']:
#             for rel in single_result['relationships']:
#                 print(
#                     f"   • {rel['entity1']['name']} --[{rel['relationship_type']}]--> {rel['entity2']['name']}")
#         else:
#             print(f"   ❌ Error: {single_result['error']}")
#         print()

#         # Test 3: Multi-entity relationships
#         print("3. Finding interactions between 'aspirin' and 'warfarin':")
#         multi_result = db_interface.extract_relationships(
#             entity_names=["Lepirudin", "warfarin"],
#             relationship_type="INTERACTS_WITH"
#         )
#         if multi_result['success']:
#             for interaction in multi_result['relationships']:
#                 print(
#                     f"   • {interaction['entity1']['name']} ↔ {interaction['entity2']['name']}")
#                 print(
#                     f"     Description: {interaction['interaction_description']}")
#         else:
#             print(f"   ❌ Error: {multi_result['error']}")
#         print()

#         print("✅ All tests completed!")

#     except Exception as e:
#         print(f"❌ Error: {e}")

#     finally:
#         if 'db_interface' in locals():
#             db_interface.close_connections()



# from crossdb import SimpleDatabaseInterface

def test_single_entity_relationships():
    config = {
        'qdrant_host': 'localhost',
        'qdrant_port': 6333,
        'collection_name': 'drug_embeddings_biobert',
        'neo4j_uri': 'bolt://localhost:7687',
        'neo4j_user': 'neo4j',
        'neo4j_password': 'your_password'
    }

    db_interface = SimpleDatabaseInterface(config)
    
    # Test drugs
    test_drugs = ["Lepirudin", "Apixaban", "Aspirin", "Warfarin", "Metformin"]
    
    for drug in test_drugs:
        print(f"\n{'='*50}")
        print(f"Testing: {drug}")
        print(f"{'='*50}")
        
        # Debug first
        debug_result = db_interface.debug_single_entity(drug)
        print(f"Debug result: {debug_result}")
        
        # Test single entity relationships
        result = db_interface.extract_relationships(
            entity_name=drug,
            relationship_type="INTERACTS_WITH",
            limit=5
        )
        
        print(f"Relationship result: {result}")
        
        if result['success']:
            print(f"✅ Found {result['relationships_count']} relationships")
            for rel in result['relationships'][:3]:  # Show first 3
                print(f"   • {rel['entity1']['name']} ↔ {rel['entity2']['name']}")
                print(f"     Type: {rel['relationship_type']}")
                if rel.get('reaction'):
                    print(f"     Reaction: {rel['reaction']['normalized_description'][:100]}...")
        else:
            print(f"❌ Error: {result['error']}")

if __name__ == "__main__":
    test_single_entity_relationships()