from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import pickle
import json
from datetime import datetime
import os

# Load BioBERT model
model_name = "dmis-lab/biobert-base-cased-v1.1"
print("🤖 Loading BioBERT model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Put model in evaluation mode
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"🔧 Using device: {device}")

def get_biobert_embedding(text):
    """Generate BioBERT embedding for a single text"""
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Use [CLS] token embedding as sentence embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # Shape: [batch_size, hidden_dim]
    return cls_embedding.squeeze().cpu().numpy()

def read_drug_names(filename="essentials/drug_names_simple.txt"):
    """Read drug names from the file"""
    
    if not os.path.exists(filename):
        print(f"❌ File '{filename}' not found!")
        return []
    
    print(f"📖 Reading drug names from '{filename}'...")
    
    drug_names = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            drug_name = line.strip()
            if drug_name:  # Skip empty lines
                drug_names.append(drug_name)
    
    print(f"✅ Loaded {len(drug_names):,} drug names")
    return drug_names

def create_drug_embeddings(drug_names, batch_size=100):
    """Create embeddings for all drug names with progress tracking"""
    
    if not drug_names:
        print("❌ No drug names to process!")
        return {}
    
    print(f"🚀 Creating embeddings for {len(drug_names):,} drugs...")
    print(f"⚙️ Batch size: {batch_size}")
    print("=" * 60)
    
    embeddings = {}
    total_drugs = len(drug_names)
    start_time = datetime.now()
    
    for i in range(0, total_drugs, batch_size):
        batch_end = min(i + batch_size, total_drugs)
        batch_drugs = drug_names[i:batch_end]
        
        print(f"🔄 Processing batch {i//batch_size + 1}: drugs {i+1:,} to {batch_end:,}")
        
        # Process each drug in the batch
        for j, drug_name in enumerate(batch_drugs):
            try:
                # Generate embedding
                embedding = get_biobert_embedding(drug_name)
                embeddings[drug_name] = embedding
                
                # Progress within batch (every 25 drugs)
                if (j + 1) % 25 == 0:
                    print(f"   📊 Processed {j+1}/{len(batch_drugs)} drugs in current batch")
                    
            except Exception as e:
                print(f"⚠️ Error processing '{drug_name}': {e}")
                continue
        
        # Overall progress
        processed = len(embeddings)
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = (total_drugs - processed) / rate if rate > 0 else 0
        
        print(f"📈 Overall Progress: {processed:,}/{total_drugs:,} ({processed/total_drugs*100:.1f}%)")
        print(f"   ⚡ Rate: {rate:.1f} drugs/sec")
        print(f"   ⏱️ Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s")
        print("-" * 40)
    
    total_time = (datetime.now() - start_time).total_seconds()
    
    print("=" * 60)
    print("✅ EMBEDDING CREATION COMPLETED!")
    print(f"⏰ Total time: {total_time:.2f} seconds")
    print(f"⚡ Average rate: {len(embeddings)/total_time:.1f} drugs/sec")
    print(f"💊 Successfully processed: {len(embeddings):,}/{total_drugs:,} drugs")
    
    if len(embeddings) < total_drugs:
        failed = total_drugs - len(embeddings)
        print(f"⚠️ Failed to process: {failed:,} drugs")
    
    return embeddings

def save_embeddings(embeddings, prefix="drug_embeddings"):
    """Save embeddings in multiple formats"""
    
    if not embeddings:
        print("❌ No embeddings to save!")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save as pickle (for fast loading in Python)
    pickle_filename = f"{prefix}_{timestamp}.pkl"
    print(f"💾 Saving embeddings as pickle: {pickle_filename}")
    
    with open(pickle_filename, 'wb') as f:
        pickle.dump(embeddings, f)
    
    # 2. Save as numpy format (more universal)
    npz_filename = f"{prefix}_{timestamp}.npz"
    print(f"💾 Saving embeddings as numpy: {npz_filename}")
    
    # Convert to arrays
    drug_names = list(embeddings.keys())
    embedding_matrix = np.array([embeddings[drug] for drug in drug_names])
    
    np.savez_compressed(npz_filename, 
                       drug_names=drug_names, 
                       embeddings=embedding_matrix)
    
    # 3. Save metadata
    metadata_filename = f"{prefix}_metadata_{timestamp}.json"
    print(f"💾 Saving metadata: {metadata_filename}")
    
    # Get embedding shape
    sample_embedding = next(iter(embeddings.values()))
    
    metadata = {
        "creation_date": datetime.now().isoformat(),
        "model_name": model_name,
        "total_drugs": len(embeddings),
        "embedding_dimension": sample_embedding.shape[0],
        "device_used": str(device),
        "sample_drugs": drug_names[:10]  # First 10 as sample
    }
    
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Embeddings saved successfully!")
    print(f"   📦 Pickle: {pickle_filename}")
    print(f"   🔢 NumPy: {npz_filename}")
    print(f"   📋 Metadata: {metadata_filename}")
    
    return pickle_filename, npz_filename, metadata_filename

def main():
    """Main function to create drug embeddings"""
    
    print("🧬 Drug Embedding Generator with BioBERT")
    print("=" * 50)
    
    # Read drug names
    drug_names = read_drug_names()
    
    if not drug_names:
        return
    
    # Show sample drug names
    print(f"\n📋 Sample drug names:")
    for i, name in enumerate(drug_names[:5]):
        print(f"   {i+1}. {name}")
    if len(drug_names) > 5:
        print(f"   ... and {len(drug_names)-5:,} more")
    
    # Create embeddings
    embeddings = create_drug_embeddings(drug_names, batch_size=50)
    
    if embeddings:
        # Save embeddings
        files = save_embeddings(embeddings)
        
        # Show sample embedding info
        sample_drug = next(iter(embeddings.keys()))
        sample_embedding = embeddings[sample_drug]
        
        print(f"\n📊 Embedding Information:")
        print(f"   🧬 Sample drug: {sample_drug}")
        print(f"   📐 Embedding shape: {sample_embedding.shape}")
        print(f"   📏 Dimension: {sample_embedding.shape[0]}")
        print(f"   🔢 Data type: {sample_embedding.dtype}")
        print(f"   📊 Value range: [{sample_embedding.min():.4f}, {sample_embedding.max():.4f}]")
        
        print(f"\n🎉 Drug embedding generation completed!")
        print(f"💊 Total drugs embedded: {len(embeddings):,}")

if __name__ == "__main__":
    main()