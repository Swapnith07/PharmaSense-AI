from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import numpy as np
import pickle
import os
import uuid
from datetime import datetime

# Configuration
QDRANT_HOST = "localhost"  # Change if you're using cloud or remote server
QDRANT_PORT = 6333
COLLECTION_NAME = "drug_embeddings_biobert"
EMBEDDING_DIM = 768  # BioBERT output dimension
BATCH_SIZE = 100  # Upload in batches to avoid payload size limit

# File path to your saved embeddings
# Replace with your actual filename
EMBEDDING_FILE = "essentials\drug_embeddings_20250701_221630.npz"

# Step 1: Connect to Qdrant
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
print(f"🔗 Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")

# Step 2: Create Collection (fix deprecation warning)
if client.collection_exists(COLLECTION_NAME):
    print(f"📦 Collection '{COLLECTION_NAME}' already exists.")
    
    # Ask user if they want to recreate
    while True:
        choice = input("🤔 Do you want to recreate the collection? (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            print(f"🗑️ Deleting existing collection...")
            client.delete_collection(COLLECTION_NAME)
            break
        elif choice in ['n', 'no']:
            print("📝 Using existing collection. Data will be upserted.")
            break
        else:
            print("❌ Please enter 'y' for yes or 'n' for no")

if not client.collection_exists(COLLECTION_NAME):
    print(f"📦 Creating new collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM, 
            distance=Distance.COSINE
        )
    )

# Step 3: Load Embeddings
print(f"📂 Loading embeddings from {EMBEDDING_FILE}...")

if EMBEDDING_FILE.endswith(".npz"):
    data = np.load(EMBEDDING_FILE, allow_pickle=True)
    drug_names = data["drug_names"]
    embeddings = data["embeddings"]
elif EMBEDDING_FILE.endswith(".pkl"):
    with open(EMBEDDING_FILE, 'rb') as f:
        embedding_dict = pickle.load(f)
    drug_names = list(embedding_dict.keys())
    embeddings = np.array([embedding_dict[name] for name in drug_names])
else:
    raise ValueError("Supported formats: .npz or .pkl")

print(f"✅ Loaded {len(drug_names):,} drug embeddings")
print(f"📐 Embedding dimension: {embeddings.shape[1]}")

# Validate embedding dimension
if embeddings.shape[1] != EMBEDDING_DIM:
    print(f"⚠️ Warning: Expected {EMBEDDING_DIM}D embeddings, got {embeddings.shape[1]}D")
    EMBEDDING_DIM = embeddings.shape[1]
    print(f"🔧 Adjusted embedding dimension to {EMBEDDING_DIM}")

# Step 4: Prepare and Upload in Batches
total_vectors = len(drug_names)
print(f"🚀 Uploading {total_vectors:,} vectors in batches of {BATCH_SIZE}...")
print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

uploaded_count = 0
start_time = datetime.now()

for batch_start in range(0, total_vectors, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, total_vectors)
    
    # Prepare batch
    batch_points = []
    for idx in range(batch_start, batch_end):
        point = PointStruct(
            id=idx,
            vector=embeddings[idx].tolist(),  # Convert numpy array to list
            payload={
                "drug_name": str(drug_names[idx]),
                "drug_id": str(idx),
                "upload_timestamp": datetime.now().isoformat()
            }
        )
        batch_points.append(point)
    
    # Upload batch
    try:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch_points
        )
        
        uploaded_count += len(batch_points)
        
        # Progress reporting
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = uploaded_count / elapsed if elapsed > 0 else 0
        remaining = (total_vectors - uploaded_count) / rate if rate > 0 else 0
        
        print(f"📈 Batch {batch_start//BATCH_SIZE + 1}: Uploaded {batch_start+1:,} to {batch_end:,}")
        print(f"   📊 Progress: {uploaded_count:,}/{total_vectors:,} ({uploaded_count/total_vectors*100:.1f}%)")
        print(f"   ⚡ Rate: {rate:.1f} vectors/sec")
        print(f"   ⏱️ Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s")
        print("-" * 40)
        
    except Exception as e:
        print(f"❌ Error uploading batch {batch_start//BATCH_SIZE + 1}: {e}")
        print(f"⚠️ Batch range: {batch_start+1} to {batch_end}")
        continue

total_time = (datetime.now() - start_time).total_seconds()

print("=" * 60)
print("✅ UPLOAD COMPLETED!")
print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🕐 Total time: {total_time:.2f} seconds")
print(f"⚡ Average rate: {uploaded_count/total_time:.1f} vectors/sec")
print(f"📊 Successfully uploaded: {uploaded_count:,}/{total_vectors:,} vectors")

# Step 5: Verify Upload
print(f"\n🔍 Verifying upload...")
collection_info = client.get_collection(COLLECTION_NAME)
vector_count = collection_info.points_count

print(f"✅ Collection '{COLLECTION_NAME}' now contains {vector_count:,} vectors")

if vector_count == total_vectors:
    print("🎉 All vectors uploaded successfully!")
else:
    print(f"⚠️ Expected {total_vectors:,} vectors, but collection has {vector_count:,}")

# Step 6: Test Search (optional)
print(f"\n🔍 Testing similarity search...")
try:
    # Search for similar drugs to the first one
    test_vector = embeddings[0].tolist()
    test_drug = drug_names[0]
    
    search_results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=test_vector,
        limit=5
    )
    
    print(f"🔍 Similar drugs to '{test_drug}':")
    for i, result in enumerate(search_results, 1):
        drug_name = result.payload['drug_name']
        score = result.score
        print(f"   {i}. {drug_name} (similarity: {score:.4f})")
    
    print("✅ Search test successful!")
    
except Exception as e:
    print(f"⚠️ Search test failed: {e}")

print(f"\n🎉 Vector database setup completed!")
print(f"📦 Collection: {COLLECTION_NAME}")
print(f"🔢 Total vectors: {vector_count:,}")
print(f"📐 Dimension: {EMBEDDING_DIM}")