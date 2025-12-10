from neo4j import GraphDatabase
import re
import time
from datetime import datetime
import json
import os

# 🔐 Neo4j connection settings
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "your_password"  

# 📊 Processing settings
# Set the number of lines to process (None for all lines)
READING_LIMIT = 500000

# 💾 Recovery settings
CHECKPOINT_FILE = "ingestion_checkpoint.json"
BATCH_SIZE = 5000  # Process in batches and save checkpoints

# Connect to Neo4j
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))


def save_checkpoint(processed_count, stats, reaction_map, reaction_counter):
    """Save current progress to checkpoint file"""
    checkpoint_data = {
        'processed_count': processed_count,
        'stats': stats,
        'reaction_map': reaction_map,
        'reaction_counter': reaction_counter,
        'timestamp': datetime.now().isoformat()
    }

    try:
        with open(CHECKPOINT_FILE, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        print(f"💾 Checkpoint saved at record {processed_count:,}")
    except Exception as e:
        print(f"⚠️ Failed to save checkpoint: {e}")


def load_checkpoint():
    """Load previous progress from checkpoint file"""
    if not os.path.exists(CHECKPOINT_FILE):
        return None

    try:
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint_data = json.load(f)

        print(f"📋 Found checkpoint from {checkpoint_data['timestamp']}")
        print(
            f"📋 Last processed: {checkpoint_data['processed_count']:,} records")
        return checkpoint_data
    except Exception as e:
        print(f"⚠️ Failed to load checkpoint: {e}")
        return None


def clear_checkpoint():
    """Remove checkpoint file after successful completion"""
    try:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
            print("🗑️ Checkpoint file cleared")
    except Exception as e:
        print(f"⚠️ Failed to clear checkpoint: {e}")


def normalize_description(desc, drug_a, drug_b):
    """Normalize reaction descriptions with placeholders"""
    normalized = re.sub(r'\b' + re.escape(drug_a) + r'\b',
                        '<drugA>', desc, flags=re.IGNORECASE)
    normalized = re.sub(r'\b' + re.escape(drug_b) + r'\b',
                        '<drugB>', normalized, flags=re.IGNORECASE)
    return normalized.strip()


def clear_database(driver):
    """Clear all nodes and relationships from Neo4j database"""
    print("🗑️  Clearing Neo4j database...")
    with driver.session() as session:
        # Delete all relationships first
        result = session.run(
            "MATCH ()-[r]-() DELETE r RETURN count(r) as deleted_relationships")
        rel_count = result.single()["deleted_relationships"]

        # Delete all nodes
        result = session.run(
            "MATCH (n) DELETE n RETURN count(n) as deleted_nodes")
        node_count = result.single()["deleted_nodes"]

        print(
            f"✅ Database cleared: {node_count:,} nodes and {rel_count:,} relationships deleted")


def check_existing_data(driver):
    """Check if database already has data"""
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN count(n) as node_count")
        node_count = result.single()["node_count"]

        result = session.run("MATCH ()-[r]-() RETURN count(r) as rel_count")
        rel_count = result.single()["rel_count"]

        return node_count, rel_count


def create_constraints(driver):
    """Create schema constraints in a separate transaction"""
    with driver.session() as session:
        try:
            session.execute_write(lambda tx: tx.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Drug) REQUIRE d.id IS UNIQUE"))
            session.execute_write(lambda tx: tx.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (r:Reaction) REQUIRE r.id IS UNIQUE"))
        except Exception as e:
            print(f"⚠️ Constraint creation warning: {e}")


def process_batch(tx, batch_data, stats, reaction_map, reaction_counter, start_index):
    """Process a batch of records with error handling"""
    # Cypher queries with duplicate checking
    drug_query = """
    MERGE (d:Drug {id: $drug_id})
    SET d.name = $drug_name
    """

    reaction_query = """
    MERGE (r:Reaction {id: $reaction_id})
    SET r.normalized_desc = $normalized_desc,
        r.example = $example_desc
    """

    interaction_check_query = """
    MATCH (a:Drug {id: $drug_a_id})-[i:INTERACTS_WITH]->(b:Drug {id: $drug_b_id})
    RETURN count(i) as exists
    """

    interaction_query = """
    MATCH (a:Drug {id: $drug_a_id}), (b:Drug {id: $drug_b_id})
    MERGE (a)-[i:INTERACTS_WITH]->(b)
    SET i.description = $description
    """

    reaction_link_check_query = """
    MATCH (d:Drug {id: $drug_id})-[:HAS_REACTION]->(r:Reaction {id: $reaction_id})
    RETURN count(*) as exists
    """

    has_reaction_query = """
    MATCH (d:Drug {id: $drug_id}), (r:Reaction {id: $reaction_id})
    MERGE (d)-[:HAS_REACTION]->(r)
    """

    drugs_seen = set()
    batch_errors = []

    for i, record in enumerate(batch_data):
        record_index = start_index + i

        try:
            drug_a_id, drug_a_name, drug_b_id, drug_b_name, description = [
                x.strip() for x in record]

            # Track drug creation
            if drug_a_id not in drugs_seen:
                drugs_seen.add(drug_a_id)
                stats['drugs_created'] += 1
            if drug_b_id not in drugs_seen:
                drugs_seen.add(drug_b_id)
                stats['drugs_created'] += 1

            # Insert Drug nodes (MERGE handles duplicates automatically)
            tx.run(drug_query, drug_id=drug_a_id, drug_name=drug_a_name)
            tx.run(drug_query, drug_id=drug_b_id, drug_name=drug_b_name)

            # Check if interaction already exists
            result = tx.run(interaction_check_query,
                            drug_a_id=drug_a_id, drug_b_id=drug_b_id)
            interaction_exists = result.single()["exists"] > 0

            if not interaction_exists:
                tx.run(interaction_query,
                       drug_a_id=drug_a_id,
                       drug_b_id=drug_b_id,
                       description=description)
                stats['interactions_created'] += 1
            else:
                stats['interactions_skipped'] += 1

            # Normalize reaction
            normalized = normalize_description(
                description, drug_a_name, drug_b_name)

            # Insert Reaction node if new
            if normalized not in reaction_map:
                reaction_id = f"R{reaction_counter:04d}"
                reaction_map[normalized] = reaction_id
                tx.run(reaction_query,
                       reaction_id=reaction_id,
                       normalized_desc=normalized,
                       example_desc=description)
                stats['reactions_created'] += 1
                reaction_counter += 1

            # Link drugs to reaction
            reaction_id = reaction_map[normalized]

            # Check and create drug A -> reaction link
            result = tx.run(reaction_link_check_query,
                            drug_id=drug_a_id, reaction_id=reaction_id)
            if result.single()["exists"] == 0:
                tx.run(has_reaction_query, drug_id=drug_a_id,
                       reaction_id=reaction_id)
                stats['drug_reaction_links'] += 1
            else:
                stats['drug_reaction_links_skipped'] += 1

            # Check and create drug B -> reaction link
            result = tx.run(reaction_link_check_query,
                            drug_id=drug_b_id, reaction_id=reaction_id)
            if result.single()["exists"] == 0:
                tx.run(has_reaction_query, drug_id=drug_b_id,
                       reaction_id=reaction_id)
                stats['drug_reaction_links'] += 1
            else:
                stats['drug_reaction_links_skipped'] += 1

            stats['processed_records'] += 1

        except Exception as e:
            error_msg = f"Error processing record {record_index}: {e}"
            batch_errors.append(error_msg)
            print(f"⚠️ {error_msg}")
            continue

    return reaction_counter, batch_errors


def import_to_neo4j_with_recovery(driver, data):
    """Import data with batch processing and recovery support"""

    # Check for existing checkpoint
    checkpoint = load_checkpoint()

    if checkpoint:
        while True:
            choice = input("🔄 Resume from checkpoint? (y/n): ").lower().strip()
            if choice in ['y', 'yes']:
                start_index = checkpoint['processed_count']
                stats = checkpoint['stats']
                reaction_map = checkpoint['reaction_map']
                reaction_counter = checkpoint['reaction_counter']
                print(f"🔄 Resuming from record {start_index:,}")
                break
            elif choice in ['n', 'no']:
                start_index = 0
                stats = {
                    'total_records': len(data),
                    'processed_records': 0,
                    'drugs_created': 0,
                    'interactions_created': 0,
                    'interactions_skipped': 0,
                    'reactions_created': 0,
                    'drug_reaction_links': 0,
                    'drug_reaction_links_skipped': 0,
                    'start_time': time.time()
                }
                reaction_map = {}
                reaction_counter = 1
                clear_checkpoint()
                print("🆕 Starting fresh import")
                break
            else:
                print("❌ Please enter 'y' for yes or 'n' for no")
    else:
        start_index = 0
        stats = {
            'total_records': len(data),
            'processed_records': 0,
            'drugs_created': 0,
            'interactions_created': 0,
            'interactions_skipped': 0,
            'reactions_created': 0,
            'drug_reaction_links': 0,
            'drug_reaction_links_skipped': 0,
            'start_time': time.time()
        }
        reaction_map = {}
        reaction_counter = 1

    print(f"🚀 Starting ingestion of {len(data) - start_index:,} records...")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    total_errors = []

    try:
        # Process data in batches
        for batch_start in range(start_index, len(data), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(data))
            batch_data = data[batch_start:batch_end]

            batch_start_time = time.time()

            try:
                with driver.session() as session:
                    reaction_counter, batch_errors = session.execute_write(
                        process_batch, batch_data, stats, reaction_map,
                        reaction_counter, batch_start
                    )

                total_errors.extend(batch_errors)

                # Save checkpoint after each successful batch
                save_checkpoint(batch_end, stats,
                                reaction_map, reaction_counter)

                # Progress reporting
                elapsed = time.time() - stats['start_time']
                rate = (batch_end - start_index) / \
                    elapsed if elapsed > 0 else 0
                remaining = (len(data) - batch_end) / rate if rate > 0 else 0

                print(
                    f"📈 Progress: {batch_end:,}/{len(data):,} ({batch_end/len(data)*100:.1f}%)")
                print(f"   ⚡ Rate: {rate:.1f} records/sec")
                print(
                    f"   ⏱️  Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s")
                print(
                    f"   📊 Drugs: {stats['drugs_created']:,} | Interactions: {stats['interactions_created']:,} (⏭️{stats['interactions_skipped']:,}) | Reactions: {stats['reactions_created']:,}")
                print(f"   ⚠️ Batch errors: {len(batch_errors)}")
                print("-" * 40)

            except Exception as e:
                print(
                    f"❌ Batch error at records {batch_start}-{batch_end}: {e}")
                print("💾 Progress saved. You can resume from this point.")
                raise

    except KeyboardInterrupt:
        print("\n⏹️ Import interrupted by user")
        print("💾 Progress has been saved. You can resume later.")
        return stats, total_errors

    except Exception as e:
        print(f"❌ Critical error during import: {e}")
        print("💾 Progress has been saved. You can resume later.")
        raise

    # Final statistics
    total_time = time.time() - stats['start_time']
    print("=" * 60)
    print("✅ INGESTION COMPLETED!")
    print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🕐 Total time: {total_time:.2f} seconds")
    print(
        f"⚡ Average rate: {stats['processed_records']/total_time:.1f} records/sec")
    print(f"📊 Final Statistics:")
    print(f"   📋 Records processed: {stats['processed_records']:,}")
    print(f"   💊 Unique drugs: {stats['drugs_created']:,}")
    print(
        f"   🔗 Drug interactions: {stats['interactions_created']:,} (⏭️ Skipped: {stats['interactions_skipped']:,})")
    print(f"   ⚡ Reaction types: {stats['reactions_created']:,}")
    print(
        f"   🔗 Drug-reaction links: {stats['drug_reaction_links']:,} (⏭️ Skipped: {stats['drug_reaction_links_skipped']:,})")
    print(f"   ⚠️ Total errors: {len(total_errors)}")
    print("=" * 60)

    # Clear checkpoint on successful completion
    clear_checkpoint()

    return stats, total_errors


# 🔁 Read TSV input file with limit
data = []
lines_read = 0
try:
    print(f"📖 Reading TSV data from 'ddi.tsv'...")
    if READING_LIMIT:
        print(f"📊 Processing limit set to: {READING_LIMIT:,} lines")
    else:
        print("📊 Processing all lines (no limit)")

    with open('ddi.tsv', 'r', encoding='utf-8') as f:
        # Read and display header for verification
        header = next(f).strip()
        print(f"📋 TSV Header: {header}")
        header_columns = header.split('\t')
        print(f"📊 Expected columns: {len(header_columns)} - {header_columns}")

        for line_num, line in enumerate(f, 1):
            if READING_LIMIT and lines_read >= READING_LIMIT:
                print(f"🛑 Reached reading limit of {READING_LIMIT:,} lines")
                break

            # Split by tab for TSV format
            parts = line.strip().split('\t')

            # Validate TSV structure
            if len(parts) >= 5:
                # Take first 5 columns: drug_a_id, drug_a_name, drug_b_id, drug_b_name, description
                data.append(parts[:5])
                lines_read += 1
            else:
                if line_num <= 10:  # Only show first 10 malformed lines to avoid spam
                    print(
                        f"⚠️  Skipping malformed line {line_num}: Expected 5+ columns, got {len(parts)}")
                continue

            # Progress for reading large files
            if lines_read % 50000 == 0:
                print(f"📖 Read {lines_read:,} valid TSV records...")

    print(f"✅ Successfully read {len(data):,} valid TSV records from ddi.tsv")

    # Show sample data for verification
    if data:
        print(f"📝 Sample record: {data[0]}")
        print(
            f"📝 Columns: drug_a_id='{data[0][0]}', drug_a_name='{data[0][1]}', drug_b_id='{data[0][2]}', drug_b_name='{data[0][3]}', description='{data[0][4][:50]}...'")

except FileNotFoundError:
    print("❌ TSV file 'ddi.tsv' not found in current directory.")
    print("📁 Please ensure 'ddi.tsv' is in the same folder as this script.")
    driver.close()
    exit(1)
except Exception as e:
    print(f"❌ Error reading TSV file 'ddi.tsv': {e}")
    import traceback
    traceback.print_exc()
    driver.close()
    exit(1)

# 🚀 Execute transactions with recovery
try:
    # Check existing data in database
    node_count, rel_count = check_existing_data(driver)

    if node_count > 0 or rel_count > 0:
        print(f"📊 Existing data found in Neo4j:")
        print(f"   🔢 Nodes: {node_count:,}")
        print(f"   🔗 Relationships: {rel_count:,}")
        print()

        while True:
            choice = input(
                "🤔 Do you want to clear the database before importing? (y/n): ").lower().strip()
            if choice in ['y', 'yes']:
                clear_database(driver)
                clear_checkpoint()  # Clear checkpoint if starting fresh
                break
            elif choice in ['n', 'no']:
                print("📝 Continuing with existing data. Duplicates will be skipped.")
                break
            else:
                print("❌ Please enter 'y' for yes or 'n' for no")
    else:
        print("✅ Database is empty, proceeding with import...")

    print("\n📌 Creating schema constraints...")
    create_constraints(driver)
    print("✅ Schema constraints created successfully!")

    if not data:
        print("⚠️ No data to process!")
    else:
        print(f"\n📌 Starting Neo4j ingestion with recovery support...")
        ingestion_start = time.time()

        final_stats, errors = import_to_neo4j_with_recovery(driver, data)

        total_ingestion_time = time.time() - ingestion_start
        print(f"\n🎉 OVERALL COMPLETION:")
        print(f"⏰ Total ingestion time: {total_ingestion_time:.2f} seconds")
        print(
            f"⚡ Overall rate: {len(data)/total_ingestion_time:.1f} records/sec")

        if errors:
            print(f"⚠️ {len(errors)} errors occurred during processing")
            with open('ingestion_errors.log', 'w') as f:
                f.write('\n'.join(errors))
            print("📝 Errors saved to 'ingestion_errors.log'")

except Exception as e:
    print(f"❌ Error during Neo4j operation: {e}")
    import traceback
    traceback.print_exc()

finally:
    driver.close()
    print("🔌 Database connection closed.")
