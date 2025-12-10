from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import traceback
from additional_chatbot import RAGChatbot, QdrantRAGDatabase

app = FastAPI(
    title="Pharmaceutical Safety Suite API",
    description="API for drug interactions, alternatives, and AI consultation",
    version="1.0.0"
)

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the medical extractor at startup with error handling
extractor = None
db_available = False

try:
    from main import MedicalTermExtractor
    extractor = MedicalTermExtractor()
    # Test database connection
    try:
        # Simple test to check if databases are working
        test_result = extractor.correct_drug_name("aspirin")
        db_available = True
        print("✅ MedicalTermExtractor loaded successfully with database access")
    except Exception as db_error:
        print(
            f"⚠️ MedicalTermExtractor loaded but database connection failed: {db_error}")
        db_available = False
except Exception as e:
    print(f"❌ Failed to load MedicalTermExtractor: {e}")
    print(f"Full error: {traceback.format_exc()}")
    extractor = None
    db_available = False

# --- Legal/Regulatory Chatbot Singleton ---
legal_chatbot = None


def get_legal_chatbot():
    global legal_chatbot
    if legal_chatbot is None:
        db = QdrantRAGDatabase(
            collection_name="drugs_cosmetics_act_rag",
            embedding_model="all-MiniLM-L6-v2"
        )
        # Use the same API key as in additional_chatbot.py or load from env
        legal_chatbot = RAGChatbot(
            db, api_key="AIzaSyBHiDJHNXqXU_q2JLq_mNma20UO0UOVq2Q")
    return legal_chatbot


class InteractionRequest(BaseModel):
    drugs: List[str]


class AlternativeRequest(BaseModel):
    drug: str
    limit: Optional[int] = 10


class ChatRequest(BaseModel):
    message: str


class NaturalLanguageRequest(BaseModel):
    query: str


class LegalChatRequest(BaseModel):
    message: str


@app.get("/")
def serve_index():
    """Serve the main index page"""
    if os.path.exists("index.html"):
        return FileResponse("index.html", media_type="text/html")
    return {"message": "Pharmaceutical Safety Suite API is running", "docs": "/docs"}


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "extractor_loaded": extractor is not None,
        "database_available": db_available,
        "message": "Pharmaceutical Safety Suite API is running"
    }


@app.post("/api/check_interactions")
async def check_interactions(req: InteractionRequest):
    """Check drug interactions for multiple drugs"""
    if not extractor:
        raise HTTPException(
            status_code=500, detail="Medical extractor not loaded")

    if len(req.drugs) < 2:
        raise HTTPException(
            status_code=400, detail="At least 2 drugs are required to check interactions")

    try:
        # Correct drug names first
        corrected_drugs = []
        for drug in req.drugs:
            try:
                corrected = extractor.correct_drug_name(drug)
                corrected_drugs.append(corrected)
            except Exception as e:
                print(f"Error correcting drug name '{drug}': {e}")
                # Use original if correction fails
                corrected_drugs.append(drug)

        # Check interactions
        try:
            interactions = extractor.check_drug_interactions(corrected_drugs)
        except Exception as e:
            print(f"Error checking interactions: {e}")
            # Return a fallback response when database is not available
            interactions = [{
                "entity1": {"name": corrected_drugs[0], "id": "unknown"},
                "entity2": {"name": corrected_drugs[1], "id": "unknown"},
                "relationship_type": "INTERACTS_WITH",
                "interaction_description": "Database temporarily unavailable. Please check with your healthcare provider.",
                "severity": "unknown"
            }]

        return {
            "success": True,
            "input_drugs": req.drugs,
            "corrected_drugs": corrected_drugs,
            "interactions": interactions,
            "total_interactions": len(interactions),
            "database_available": db_available
        }
    except Exception as e:
        print(f"Error in check_interactions: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, detail=f"Error checking interactions: {str(e)}")


@app.post("/api/find_alternatives")
async def find_alternatives(req: AlternativeRequest):
    """Find alternative drugs for a given drug"""
    if not extractor:
        raise HTTPException(
            status_code=500, detail="Medical extractor not loaded")

    if not req.drug.strip():
        raise HTTPException(
            status_code=400, detail="Drug name cannot be empty")

    try:
        # Correct drug name
        try:
            corrected_drug = extractor.correct_drug_name(req.drug)
        except Exception as e:
            print(f"Error correcting drug name: {e}")
            corrected_drug = req.drug

        # Find alternatives
        try:
            alternatives = extractor.find_drug_alternatives(corrected_drug)
        except Exception as e:
            print(f"Error finding alternatives: {e}")
            alternatives = [{
                "entity_name": "Database temporarily unavailable",
                "entity_id": "unknown",
                "similarity_score": 0.0,
                "payload": {"message": "Please check with your healthcare provider for alternatives."}
            }]

        return {
            "success": True,
            "input_drug": req.drug,
            "corrected_drug": corrected_drug,
            "alternatives": alternatives,
            "total_alternatives": len(alternatives),
            "database_available": db_available
        }
    except Exception as e:
        print(f"Error in find_alternatives: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error finding alternatives: {str(e)}")


@app.post("/api/ai_consultant")
async def ai_consultant(req: ChatRequest):
    """AI consultant for natural language queries"""
    if not extractor:
        raise HTTPException(
            status_code=500, detail="Medical extractor not loaded")

    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        # Process the query using the same method as Streamlit
        result = extractor.process_query(req.message)

        # Add database availability info
        if isinstance(result, dict):
            result['database_available'] = db_available

        return result
    except Exception as e:
        print(f"Error in ai_consultant: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        return {
            "success": False,
            "error": str(e),
            "ai_response": "I'm having trouble processing your query right now. Please try again or contact your healthcare provider.",
            "database_available": db_available
        }


@app.post("/api/process_natural_language")
async def process_natural_language(req: NaturalLanguageRequest):
    """Process natural language queries (same as ai_consultant but different endpoint)"""
    if not extractor:
        raise HTTPException(
            status_code=500, detail="Medical extractor not loaded")

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        # Process the natural language query
        result = extractor.process_query(req.query)

        # Add database availability info
        if isinstance(result, dict):
            result['database_available'] = db_available

        return result
    except Exception as e:
        print(f"Error in process_natural_language: {e}")
        return {
            "success": False,
            "error": str(e),
            "ai_response": "I'm having trouble processing your query right now. Please try again.",
            "database_available": db_available
        }


@app.post("/api/correct_drug_name")
async def correct_drug_name(req: AlternativeRequest):
    """Correct/standardize a drug name"""
    if not extractor:
        raise HTTPException(
            status_code=500, detail="Medical extractor not loaded")

    if not req.drug.strip():
        raise HTTPException(
            status_code=400, detail="Drug name cannot be empty")

    try:
        corrected_drug = extractor.correct_drug_name(req.drug)

        return {
            "success": True,
            "input_drug": req.drug,
            "corrected_drug": corrected_drug,
            "was_corrected": req.drug != corrected_drug,
            "database_available": db_available
        }
    except Exception as e:
        print(f"Error in correct_drug_name: {e}")
        return {
            "success": False,
            "input_drug": req.drug,
            "corrected_drug": req.drug,
            "was_corrected": False,
            "error": str(e),
            "database_available": db_available
        }


@app.get("/api/popular_drugs")
async def get_popular_drugs():
    """Get list of popular drugs for quick searches"""
    return {
        "popular_drugs": [
            "Aspirin",
            "Metformin",
            "Lepirudin",
            "Prazosin",
            "Warfarin",
            "Ibuprofen",
            "Acetaminophen",
            "Apixaban",
            "Rivaroxaban"
        ]
    }


@app.get("/api/drug_examples")
async def get_drug_examples():
    """Get example drug combinations for interaction checking"""
    return {
        "examples": [
            ["Lepirudin", "Apixaban"],
            ["Aspirin", "Warfarin"],
            ["Metformin", "Insulin"],
            ["Prazosin", "Doxazosin"],
            ["Ibuprofen", "Aspirin"]
        ]
    }


@app.get("/api/status")
async def get_system_status():
    """Get detailed system status"""
    return {
        "extractor_loaded": extractor is not None,
        "database_available": db_available,
        "status": "running",
        "message": "Pharmaceutical Safety Suite API is operational"
    }


@app.post("/api/legal_chatbot")
async def legal_chatbot_endpoint(req: LegalChatRequest):
    """Legal/Regulatory chatbot for Drugs and Cosmetics Act Q&A"""
    bot = get_legal_chatbot()
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    try:
        answer = bot.chat(req.message)
        return {"success": True, "ai_response": answer}
    except Exception as e:
        print(f"Error in legal_chatbot: {e}")
        return {"success": False, "error": str(e), "ai_response": "Error processing your query. Please try again."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
