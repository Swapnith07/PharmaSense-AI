from typing import List, Dict
from crossdb import SimpleDatabaseInterface
from agents import PharmaceuticalAgentSystem


class MedicalTermExtractor:
    def __init__(self):
        self.agent_system = PharmaceuticalAgentSystem()

        db_config = {
            'qdrant_host': 'localhost',
            'qdrant_port': 6333,
            'collection_name': 'drug_embeddings_biobert',
            'neo4j_uri': 'bolt://localhost:7687',
            'neo4j_user': 'neo4j',
            'neo4j_password': 'your_password'
        }

        self.db_interface = SimpleDatabaseInterface(db_config)

    def extract_medical_terms(self, query: str) -> List[str]:
        """Extract drug names using AI agent"""
        return self.agent_system.extract_drugs(query)

    def correct_drug_name(self, query_drug: str, score_threshold: float = 0.5) -> str:
        """Correct drug name using vector similarity search"""
        try:
            results = self.db_interface.find_similar_entities(
                entity_name=query_drug, limit=1, score_threshold=score_threshold
            )

            if results.get('success', False) and results['results']:
                top_result = results['results'][0]
                corrected_name = top_result['entity_name']
                similarity = top_result['similarity_score']

                if similarity >= score_threshold:
                    print(
                        f"Corrected '{query_drug}' → '{corrected_name}' ({similarity:.2f})")
                    return corrected_name

            return query_drug
        except Exception as e:
            print(f"Error correcting '{query_drug}': {e}")
            return query_drug

    def check_drug_interactions(self, drug_names: List[str]) -> List[Dict]:
        """Real-Time Prescription Safety Checker - returns ALL interactions"""
        try:
            if len(drug_names) < 2:
                return []

            # Get all possible interactions between drugs
            result = self.db_interface.extract_relationships(
                entity_names=drug_names,
                relationship_type="INTERACTS_WITH",
                limit=1000  # Get all interactions
            )

            if result.get('success', False):
                return result.get('relationships', [])
            return []

        except Exception as e:
            print(f"Error checking interactions: {e}")
            return []

    def find_drug_alternatives(self, drug_name: str) -> List[Dict]:
        """Smart Drug Substitution Engine - returns top 5 similar drugs (excluding the input drug)"""
        try:
            result = self.db_interface.find_similar_entities(
                entity_name=drug_name,
                limit=15,  # Get more to filter out the input drug
                score_threshold=0.3
            )

            if result.get('success', False):
                alternatives = result.get('results', [])
                
                # Filter out the input drug itself (exact matches and high similarity)
                filtered_alternatives = []
                input_drug_lower = drug_name.lower().strip()
                
                for drug in alternatives:
                    drug_name_db = drug.get('entity_name', '').lower().strip()
                    similarity = drug.get('similarity_score', 0)
                    
                    # Skip if it's the same drug (exact match or very high similarity >0.98)
                    if (drug_name_db != input_drug_lower and 
                        similarity < 0.98 and 
                        len(filtered_alternatives) < 5):
                        filtered_alternatives.append(drug)
                
                return filtered_alternatives[:5]  # Return only top 5
            return []

        except Exception as e:
            print(f"Error finding alternatives for '{drug_name}': {e}")
            return []

    def process_query(self, query: str) -> Dict:
        """Enhanced main method with intent validation"""
        try:
            # Extract and correct drug names
            raw_drugs = self.agent_system.extract_drugs(query)
            corrected_drugs = [self.correct_drug_name(drug) for drug in raw_drugs]

            # Classify intent with enhanced validation
            intent = self.agent_system.classify_intent(query, corrected_drugs)
            
            # Additional intent validation logic
            query_lower = query.lower()
            
            # Override intent if clear interaction patterns detected
            if (len(corrected_drugs) >= 2 and 
                any(pattern in query_lower for pattern in ['can i take', 'safe to take', 'take with', 'with']) and
                intent != "check_interaction"):
                
                print(f"🔧 Intent corrected: {intent} → check_interaction")
                intent = "check_interaction"
            
            # Process based on intent
            db_results = {}

            if intent == "check_interaction" and len(corrected_drugs) >= 2:
                interactions = self.check_drug_interactions(corrected_drugs)
                db_results = {'interactions': interactions}

            elif intent == "find_similar" and corrected_drugs:
                similar_drugs = self.find_drug_alternatives(corrected_drugs[0])
                db_results = {'similar_drugs': similar_drugs}

            elif intent == "general_query":
                # For general queries, try to get both interactions and similar drugs if drugs are present
                if corrected_drugs:
                    if len(corrected_drugs) >= 2:
                        interactions = self.check_drug_interactions(corrected_drugs)
                        db_results['interactions'] = interactions

                    # Always try to get similar drugs for the first drug mentioned
                    similar_drugs = self.find_drug_alternatives(corrected_drugs[0])
                    db_results['similar_drugs'] = similar_drugs
                else:
                    # No drugs mentioned - general pharmaceutical info
                    db_results = {'general_info': True}

            # Generate enhanced AI response
            ai_response = self.agent_system.generate_response(intent, corrected_drugs, db_results)

            return {
                'success': True,
                'raw_drugs': raw_drugs,
                'corrected_drugs': corrected_drugs,
                'intent': intent,
                'database_results': db_results,
                'ai_response': ai_response
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'ai_response': "Error processing your query. Please try again."
            }
