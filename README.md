# 🏥 PharmaSense AI - Pharmaceutical Safety Intelligence System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)
![Neo4j](https://img.shields.io/badge/Neo4j-5.0+-red.svg)
![Qdrant](https://img.shields.io/badge/Qdrant-Latest-purple.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**An intelligent multi-agent system for drug interaction detection and pharmaceutical information retrieval.**

[Features](#features) • [Architecture](#architecture) • [Quick Start](#quick-start) • [API](#api-endpoints) • [Usage](#usage-examples)

</div>

---

## 📋 Overview

PharmaSense AI is a sophisticated pharmaceutical safety system that combines **multi-agent NLP**, **hybrid databases**, and **medical safety protocols** to provide accurate drug interaction information and pharmaceutical guidance.

### Key Capabilities

- ✅ **Drug Interaction Checking** - Analyze interactions between multiple medications
- ✅ **Alternative Drug Finding** - Discover therapeutically similar medications
- ✅ **Legal/Regulatory Queries** - Access pharmaceutical regulations and compliance information
- ✅ **Natural Language Processing** - Understand conversational pharmaceutical queries
- ✅ **Safety-First Design** - Mandatory medical disclaimers and healthcare provider recommendations

---

## 🎯 Features

### Core Functionality

| Feature | Description |
|---------|-------------|
| **Drug Interaction Detection** | Multi-drug interaction analysis with severity classification |
| **Intent Classification** | Accurately understand user intent (check_interaction, find_similar, general_query) |
| **Drug Name Correction** | Intelligent spelling correction for misspelled drug names |
| **Information Accuracy** | Factually correct pharmaceutical data from trusted sources |
| **Safety Considerations** | Consistent medical disclaimers and healthcare provider recommendations |
| **Regulatory Information** | RAG-based retrieval from pharmaceutical regulations |

### Advanced Features

- 🤖 **Multi-Agent Architecture** - Specialized agents for NER, intent classification, and response generation
- 🔗 **Hybrid Database System** - Neo4j for relationships + Qdrant for semantic search
- 📚 **RAG-Enhanced Responses** - Retrieval-Augmented Generation for regulatory queries
- 🧬 **BioBERT Embeddings** - Medical-grade embeddings for pharmaceutical entity recognition
- 🛡️ **Privacy-First Design** - Stateless processing, no user data retention

---

## 🏗️ System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                       │
│              (Web UI - index.html)                      │
└─────────────────────────────┬───────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────┐
│              FastAPI Server (server.py)                 │
│         /check_interactions, /find_alternatives         │
└─────────────────────────────┬───────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────┐
│         Main Orchestrator (main.py)                     │
│    Coordinates Multi-Agent Processing Pipeline         │
└──────┬──────────────────┬──────────────────┬────────────┘
       │                  │                  │
   ┌───▼────┐         ┌───▼────┐        ┌───▼────┐
   │ NER    │         │ Intent │        │Response│
   │ Agent  │         │ Agent  │        │ Agent  │
   │(agents)│         │(agents)│        │(agents)│
   └───┬────┘         └───┬────┘        └───┬────┘
       │                  │                  │
       └──────────────┬───┴──────────────┬───┘
                      │                  │
              ┌───────▼────────┐    ┌────▼──────────┐
              │ Database       │    │ Database      │
              │ Interface      │    │ Interface     │
              └───────┬────────┘    └────┬──────────┘
                      │                  │
         ┌────────────▼────────┐  ┌──────▼─────────┐
         │   Neo4j Graph DB    │  │ Qdrant Vector  │
         │ (Drug Interactions) │  │ DB (Embeddings)│
         └─────────────────────┘  └────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker & Docker Compose (recommended)
- 8GB RAM minimum
- Neo4j 5.0+
- Qdrant (vector database)

### Installation

#### Option 1: Docker Compose (Recommended)

```bash
# Clone repository
git clone https://github.com/Swapnith07/PharmaSense-AI.git
cd PharmaSense-AI

# Start all services
docker-compose up -d

# API available at: http://localhost:8000
```

#### Option 2: Manual Setup

```bash
# Clone repository
git clone https://github.com/Swapnith07/PharmaSense-AI.git
cd PharmaSense-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Neo4j (Docker)
docker run --restart always -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/StrongPass123 \
  -v "$(pwd)/neo4j/data:/data" \
  neo4j:latest

# Start Qdrant (Docker)
docker run -p 6333:6333 -p 6334:6334 \
  -v "$(pwd)/qdrant_storage:/qdrant/storage" \
  qdrant/qdrant

# Start FastAPI server
uvicorn server:app --reload
```

### First Query

```bash
# Check drug interaction
curl -X POST http://localhost:8000/api/check_interactions \
  -H "Content-Type: application/json" \
  -d '{"message": "Can I take aspirin with warfarin?"}'

# Response:
# {
#   "success": true,
#   "ai_response": "⚠️ MAJOR INTERACTION WARNING...",
#   "intent": "check_interaction"
# }
```

---

## 📁 Project Structure

```
PharmaSense-AI/
├── 📄 main.py                 # Core orchestration logic
├── 🌐 server.py               # FastAPI application
├── 🤖 agents.py               # Multi-agent system (NER, Intent, Response)
├── 💾 graphdb.py              # Neo4j database interface
├── 🔍 vectordb.py             # Qdrant vector database interface
├── 🔗 crossdb.py              # Unified database abstraction
├── 📚 additional_chatbot.py    # Legal/Regulatory RAG chatbot
├── 🧬 embeddings.py           # BioBERT embedding generation
├── 🎨 index.html              # Web UI (3-tab interface)
│
├── 📦 requirements.txt         # Python dependencies
├── ⚙️ config.json              # Configuration settings
│
├── 📊 essentials/
│   ├── ddi.tsv                # 500K+ drug-drug interactions
│   ├── drug_embeddings_*.npz  # BioBERT embeddings (4000+ drugs)
│   └── db.txt                 # Database connection commands
│
├── 🧠 models/
│   └── all-MiniLM-L6-v2/      # Sentence-transformers model
│
└── 🗄️ qdrant_storage/         # Vector database storage
    └── collections/
```

---

## 🔌 API Endpoints

### Drug Interaction Checker

```bash
POST /api/check_interactions
Content-Type: application/json

{
  "message": "Can I take aspirin with warfarin?"
}

Response:
{
  "success": true,
  "ai_response": "⚠️ MAJOR INTERACTION WARNING...",
  "intent": "check_interaction",
  "safety_level": "MAJOR_INTERACTION"
}
```

### Alternative Drug Finder

```bash
POST /api/find_alternatives
Content-Type: application/json

{
  "message": "What drugs are similar to aspirin?"
}

Response:
{
  "success": true,
  "ai_response": "Similar alternatives include...",
  "intent": "find_similar"
}
```

### General AI Consultant

```bash
POST /api/ai_consultant
Content-Type: application/json

{
  "message": "What is ibuprofen used for?"
}

Response:
{
  "success": true,
  "ai_response": "Ibuprofen is...",
  "intent": "general_query"
}
```

### Legal/Regulatory Chatbot

```bash
POST /api/legal_chatbot
Content-Type: application/json

{
  "message": "What are labeling requirements?"
}

Response:
{
  "success": true,
  "ai_response": "According to regulations..."
}
```

---

## 💡 Usage Examples

### Example 1: Drug Interaction Query

```python
query = "Can I take ibuprofen with aspirin?"

# System Response:
# ⚠️ MODERATE INTERACTION WARNING
# Taking ibuprofen with aspirin can increase the risk of 
# bleeding and gastric irritation. Both medications are NSAIDs 
# and can cause additive side effects.
# 
# Consider using only one NSAID at a time and consult with 
# a healthcare professional for pain management alternatives.
```

### Example 2: Alternative Drug Search

```python
query = "What drugs are similar to aspirin?"

# System Response:
# Similar alternatives to aspirin include:
# • Ibuprofen (Advil, Motrin)
# • Naproxen (Aleve)
# • Acetaminophen (Tylenol)
# • Celecoxib (Celebrex)
#
# Each has different benefits and risks. Consult your 
# healthcare provider for recommendations.
```

### Example 3: Drug Information Query

```python
query = "What is metformin used for?"

# System Response:
# Metformin is primarily used to treat:
# • Type 2 diabetes mellitus
# • Prediabetes
# • Polycystic ovary syndrome (PCOS)
#
# It works by reducing glucose production in the liver 
# and improving insulin sensitivity...
```

---

## 🔐 Security & Privacy

### Privacy Protections

✅ **No Data Retention** - Stateless processing
✅ **No Session State** - Each request is independent
✅ **Medical Disclaimers** - Explicit in every response
✅ **Scope Limitation** - Only pharmaceutical information
✅ **Safe Degradation** - Transparent error handling

### Medical Safety

✅ **Safety-First Design** - Medical disclaimers always included
✅ **Severity Classifications** - SAFE, CAUTION, MAJOR_INTERACTION
✅ **Healthcare Provider Referral** - Always recommended
✅ **No Medical Advice** - Information only, decisions deferred to professionals

---

## ⚙️ Configuration

Edit `config.json` to customize:

```json
{
  "gemini_api_key": "your-api-key",
  "neo4j_uri": "bolt://localhost:7687",
  "neo4j_user": "neo4j",
  "neo4j_password": "StrongPass123",
  "qdrant_host": "localhost",
  "qdrant_port": 6333,
  "similarity_threshold": 0.35,
  "max_results": 5
}
```

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Submit a pull request

### Development Setup

```bash
git clone https://github.com/Swapnith07/PharmaSense-AI.git
cd PharmaSense-AI

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

docker-compose up -d
uvicorn server:app --reload
```

---

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## ⚖️ Legal Disclaimer

**PharmaSense AI is for educational and informational purposes only.**

- ⚠️ NOT a substitute for professional medical advice
- ⚠️ Always consult with a licensed healthcare provider
- ⚠️ Information may change - verify with current sources

**Use at your own risk. The authors assume no liability for misuse or medical decisions made based on this system.**-

## 📞 Contact & Support

- 📧 Email: swapnith07@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/Swapnith07/PharmaSense-AI/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/Swapnith07/PharmaSense-AI/discussions)

---

## 🙏 Acknowledgments

- BioBERT embeddings from [DMIS-Lab](https://github.com/dmis-lab/biobert)
- Gemini AI for advanced language understanding
- Neo4j and Qdrant communities

---

<div align="center">

**Made with ❤️ for pharmaceutical safety**

⭐ If this project helps you, please consider giving it a star!

</div>
