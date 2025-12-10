from neo4j import GraphDatabase

# 🔐 Neo4j connection settings
URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "your_password"  # Replace with your actual password

# Connect to Neo4j
driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def extract_drug_names_simple():
    """Extract only drug names, one per line"""
    
    print("💊 Extracting drug names...")
    
    with driver.session() as session:
        # Get drug names only
        result = session.run("MATCH (d:Drug) RETURN d.name as name ORDER BY d.name")
        
        drug_names = []
        for record in result:
            name = record["name"]
            if name:  # Only add if name exists
                drug_names.append(name)
        
        print(f"✅ Found {len(drug_names):,} drug names")
        
        # Save to simple text file
        filename = "drug_names_simple.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            for name in drug_names:
                f.write(f"{name}\n")
        
        print(f"💾 Saved to: {filename}")
        
        # Show first few names
        print(f"\n📋 First 5 drug names:")
        for i, name in enumerate(drug_names[:5]):
            print(f"   {name}")
    
    driver.close()

if __name__ == "__main__":
    extract_drug_names_simple()