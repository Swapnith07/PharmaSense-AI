from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from typing import List

# Load model once
model_name = "pruas/BENT-PubMedBERT-NER-Chemical"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

def extract_drugs_biobert(text: str) -> List[str]:
    """
    Extract drug/chemical names using PubMedBERT-based chemical NER.
    """
    try:
        entities = ner_pipeline(text)
        drugs = []
        for ent in entities:
            # entity_group may be something like 'DRUG', 'CHEMICAL', etc.
            label = ent.get("entity_group", ent.get("entity"))
            if label and label.lower() in ["drug", "chemical"]:
                name = ent['word'].strip()
                if name and name not in drugs:
                    drugs.append(name)
        return drugs
    except Exception as e:
        print(f"⚠️ NER error: {e}")
        return []

# Test
if __name__ == "__main__":
    for query in [
        "Can I take aspirin with warfarin?",
        "Side effects of metformin and insulin?",
        "Combine Lepirudin with Apixaban?"
    ]:
        print(query, extract_drugs_biobert(query))
