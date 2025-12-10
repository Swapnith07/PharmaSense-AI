import re
import uuid
import logging
import os
import shutil
from typing import List, Dict, Optional
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from agno.agent import Agent
from agno.models.google import Gemini

logging.basicConfig(level=logging.INFO)


def clear_sentence_transformer_cache():
    """Clear the sentence transformer cache to force re-download"""
    try:
        import sentence_transformers
        cache_dir = os.path.join(os.path.expanduser(
            "~"), ".cache", "huggingface", "hub")
        model_cache_dir = os.path.join(
            cache_dir, "models--sentence-transformers--all-MiniLM-L6-v2")

        if os.path.exists(model_cache_dir):
            shutil.rmtree(model_cache_dir)
            logging.info("✅ Cleared sentence transformer cache")
            print("✅ Cleared corrupted model cache")
        else:
            logging.info("ℹ️ No cache directory found")
    except Exception as e:
        logging.error(f"❌ Error clearing cache: {e}")


class QdrantRAGDatabase:
    """
    Handles all Qdrant database operations for RAG system,
    with section-level chunking and rich metadata.
    """

    def __init__(
        self,
        collection_name: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
    ):
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Set cache directory to current directory
        current_dir = os.getcwd()
        model_cache_dir = os.path.join(current_dir, "models", embedding_model)

        # Create models directory if it doesn't exist
        os.makedirs(model_cache_dir, exist_ok=True)

        # Try to load the model with custom cache directory
        try:
            self.model = SentenceTransformer(
                embedding_model, cache_folder=model_cache_dir)
            print(f"✅ Model loaded successfully from: {model_cache_dir}")
        except Exception as e:
            logging.error(f"❌ Failed to load model: {e}")
            print("🔧 Attempting to fix corrupted model cache...")

            # Clear both default cache and local cache
            clear_sentence_transformer_cache()
            if os.path.exists(model_cache_dir):
                shutil.rmtree(model_cache_dir)
                os.makedirs(model_cache_dir, exist_ok=True)

            print(f"⏳ Re-downloading model to: {model_cache_dir}")
            print("   This may take a moment...")
            self.model = SentenceTransformer(
                embedding_model, cache_folder=model_cache_dir)
            print("✅ Model downloaded and loaded successfully!")

        self.client = QdrantClient(qdrant_host, port=qdrant_port)
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            collections = [
                c.name for c in self.client.get_collections().collections]
            if self.collection_name not in collections:
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.model.get_sentence_embedding_dimension(),
                        distance=Distance.COSINE
                    )
                )
                logging.info(f"✅ Created collection: {self.collection_name}")
        except Exception as e:
            logging.error(f"❌ Error creating collection: {e}")

    def check_if_pdf_loaded(self, pdf_path: str) -> bool:
        """Check if PDF is already loaded in the database"""
        try:
            # Search for any document with this source file
            result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter={
                    "must": [
                        {
                            "key": "source_file",
                            "match": {"value": pdf_path}
                        }
                    ]
                },
                limit=1
            )

            if result[0]:  # If any points found
                logging.info(f"📚 PDF already loaded: {pdf_path}")
                return True
            return False
        except Exception as e:
            logging.error(f"❌ Error checking PDF status: {e}")
            return False

    def get_collection_count(self) -> int:
        """Get total number of documents in collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count
        except Exception as e:
            logging.error(f"❌ Error getting collection count: {e}")
            return 0

    def parse_pdf_to_chunks(self, pdf_path: str, max_chunk_length: int = 800) -> List[Dict]:
        """
        Parse PDF and extract structured chunks at section level.
        If a section is long, split into smaller chunks.
        """
        try:
            reader = PdfReader(pdf_path)
            chunks = []
            chapter = section = section_title = None
            chunk_id = 0

            chapter_pattern = re.compile(
                r'^CHAPTER\s+[A-Z]+\b.*', re.IGNORECASE)
            section_pattern = re.compile(
                r'^(\d+[A-Z]?)\.\s+(.+)', re.IGNORECASE)

            buffer = []
            buffer_section = buffer_section_title = buffer_chapter = None
            buffer_page = 1

            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if not text:
                    continue
                lines = text.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if chapter_pattern.match(line):
                        chapter = line
                        continue
                    sec_match = section_pattern.match(line)
                    if sec_match:
                        # Save previous section as chunk
                        if buffer:
                            self._split_and_append_chunk(
                                chunks, buffer, buffer_chapter, buffer_section, buffer_section_title,
                                buffer_page, chunk_id, pdf_path, max_chunk_length
                            )
                            chunk_id += 1
                            buffer = []
                        section = sec_match.group(1)
                        section_title = sec_match.group(2).strip()
                        buffer_section = section
                        buffer_section_title = section_title
                        buffer_chapter = chapter
                        buffer_page = page_num
                        buffer.append(f"Section {section}: {section_title}")
                        continue
                    buffer.append(line)
                # End of page
            if buffer:
                self._split_and_append_chunk(
                    chunks, buffer, buffer_chapter, buffer_section, buffer_section_title,
                    buffer_page, chunk_id, pdf_path, max_chunk_length
                )
            logging.info(f"📄 Parsed {len(chunks)} chunks from PDF")
            return chunks
        except Exception as e:
            logging.error(f"❌ Error parsing PDF: {e}")
            return []

    def _split_and_append_chunk(
        self, chunks, buffer, chapter, section, section_title, page_num, chunk_id, pdf_path, max_chunk_length
    ):
        """
        Split large section buffer into smaller chunks if needed, preserving metadata.
        """
        text = "\n".join(buffer)
        paragraphs = text.split('\n\n')
        current = []
        current_len = 0
        sub_chunk = 0
        for para in paragraphs:
            if current_len + len(para) > max_chunk_length and current:
                # Save current chunk
                chunks.append(self._create_chunk(
                    chapter, section, section_title, page_num,
                    f"{chunk_id}_{sub_chunk}", "\n\n".join(current), pdf_path
                ))
                current = []
                current_len = 0
                sub_chunk += 1
            current.append(para)
            current_len += len(para)
        if current:
            chunks.append(self._create_chunk(
                chapter, section, section_title, page_num,
                f"{chunk_id}_{sub_chunk}", "\n\n".join(current), pdf_path
            ))

    def _create_chunk(self, chapter, section, section_title, page_num, chunk_id, text, pdf_path):
        """Create a structured chunk"""
        return {
            "chapter_title": chapter,
            "section_number": section,
            "section_title": section_title,
            "page_number": page_num,
            "chunk_id": str(chunk_id),
            "text": text,
            "source_file": pdf_path
        }

    def add_chunks_to_db(self, chunks: List[Dict]):
        """Add parsed PDF chunks to database"""
        try:
            points = []
            for chunk in chunks:
                if chunk["text"].strip():  # Only add non-empty chunks
                    embedding = self.model.encode(
                        chunk["text"], convert_to_numpy=True)
                    points.append(PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding.tolist(),
                        payload=chunk
                    ))
            if points:
                self.client.upsert(
                    collection_name=self.collection_name, points=points)
                logging.info(f"✅ Added {len(points)} chunks to database")
        except Exception as e:
            logging.error(f"❌ Error adding chunks to database: {e}")

    def add_texts_to_db(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """Add plain text documents with optional metadata to database"""
        try:
            points = []
            for i, text in enumerate(texts):
                meta = metadatas[i] if metadatas and i < len(metadatas) else {}
                embedding = self.model.encode(text, convert_to_numpy=True)
                payload = {"text": text, **meta}
                points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload=payload
                ))
            if points:
                self.client.upsert(
                    collection_name=self.collection_name, points=points)
                logging.info(
                    f"✅ Added {len(points)} text documents to database")
        except Exception as e:
            logging.error(f"❌ Error adding texts to database: {e}")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant chunks, return structured context"""
        try:
            query_vec = self.model.encode(
                query, convert_to_numpy=True).tolist()
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vec,
                limit=top_k,
                with_payload=True
            )
            formatted_results = []
            for hit in results:
                payload = hit.payload
                formatted_results.append({
                    "chapter_title": payload.get("chapter_title"),
                    "section_number": payload.get("section_number"),
                    "section_title": payload.get("section_title"),
                    "page_number": payload.get("page_number"),
                    "chunk_id": payload.get("chunk_id"),
                    "text": payload.get("text", ""),
                    "score": hit.score,
                    "source_file": payload.get("source_file"),
                })
            return formatted_results
        except Exception as e:
            logging.error(f"❌ Error searching database: {e}")
            return []

    def build_prompt(self, query: str, contexts: List[Dict]) -> str:
        """Build a clear, context-rich prompt for the LLM"""
        context_strs = []
        for i, ctx in enumerate(contexts, 1):
            context_strs.append(
                f"Context {i}:\n"
                f"Chapter: {ctx['chapter_title']}\n"
                f"Section {ctx['section_number']}: {ctx['section_title']}\n"
                f"Page: {ctx['page_number']}\n"
                f"Text:\n{ctx['text']}\n"
            )
        context_block = "\n---\n".join(context_strs)
        prompt = (
            f"You are a legal assistant. Use only the provided legal context to answer the question as precisely as possible.\n"
            f"Context:\n{context_block}\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        return prompt


class RAGChatbot:
    """
    RAG Chatbot that uses Qdrant database and Gemini for responses
    """

    def __init__(self, qdrant_db: QdrantRAGDatabase, api_key: Optional[str] = None):
        self.db = qdrant_db
        self.agent = Agent(
            model=Gemini(id="gemini-1.5-flash", api_key=api_key),
            markdown=True,
            instructions=[
                "You are a helpful RAG assistant.",
                "Use the provided context to answer questions accurately.",
                "If the context doesn't contain relevant information, say so clearly.",
                "Always cite the source when possible.",
                "Be concise but comprehensive in your responses."
            ]
        )

    def load_pdf_knowledge(self, pdf_path: str, force_reload: bool = False):
        """Load knowledge from PDF file with duplicate check"""
        logging.info(f"📚 Loading knowledge from: {pdf_path}")

        # Check if PDF is already loaded (unless force reload)
        if not force_reload and self.db.check_if_pdf_loaded(pdf_path):
            current_count = self.db.get_collection_count()
            logging.info(
                f"✅ PDF already loaded. Current documents in DB: {current_count}")
            print(
                f"✅ PDF already loaded. Current documents in DB: {current_count}")
            return

        # Parse and load PDF
        chunks = self.db.parse_pdf_to_chunks(pdf_path)
        if chunks:
            if force_reload:
                logging.info("🔄 Force reloading - clearing existing data...")
                # Optionally clear existing data for this PDF
                # self.db.clear_pdf_data(pdf_path)

            self.db.add_chunks_to_db(chunks)
            current_count = self.db.get_collection_count()
            logging.info(
                f"✅ Successfully loaded PDF knowledge. Total documents: {current_count}")
            print(
                f"✅ Successfully loaded PDF knowledge. Total documents: {current_count}")
        else:
            logging.error(f"❌ Failed to load PDF knowledge")
            print(f"❌ Failed to load PDF knowledge")

    def load_text_knowledge(self, texts: List[str], metadata: List[Dict] = None):
        """Load knowledge from text list"""
        logging.info(f"📚 Loading {len(texts)} text documents...")
        self.db.add_texts_to_db(texts, metadata)
        logging.info(f"✅ Successfully loaded text knowledge")

    def chat(self, query: str, top_k: int = 5) -> str:
        """Chat with context retrieval"""
        try:
            results = self.db.search(query, top_k=top_k)
            if not results:
                return "I don't have relevant information to answer your question."
            context_parts = []
            for i, result in enumerate(results, 1):
                context_parts.append(f"Context {i}:\n{result['text']}")
            context = "\n\n".join(context_parts)
            prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Answer:"""
            response = self.agent.run(prompt)
            return response.content
        except Exception as e:
            return f"❌ Error processing query: {e}"

    def print_chat(self, query: str, top_k: int = 5):
        """Chat with streaming response"""
        try:
            results = self.db.search(query, top_k=top_k)
            if not results:
                print("I don't have relevant information to answer your question.")
                return
            context_parts = []
            for i, result in enumerate(results, 1):
                context_parts.append(f"Context {i}:\n{result['text']}")
            context = "\n\n".join(context_parts)
            prompt = f"""Based on the following context, please answer the question.

Context:
{context}

Question: {query}

Answer:"""
            self.agent.print_response(prompt, stream=True)
        except Exception as e:
            print(f"❌ Error processing query: {e}")


def main():
    """Main function to run the RAG chatbot"""
    print("🤖 Initializing RAG Chatbot...")

    # Initialize Qdrant database
    db = QdrantRAGDatabase(
        collection_name="drugs_cosmetics_act_rag",
        embedding_model="all-MiniLM-L6-v2"
    )

    # Initialize chatbot
    chatbot = RAGChatbot(db, api_key="AIzaSyCBbU7Ss6Qw4kj7lIyN4WjjeZTALlP9wMc")

    # Load PDF knowledge (with duplicate check)
    pdf_path = "essentials\Drugs and Cosmetics Act, 1940.pdf"
    chatbot.load_pdf_knowledge(pdf_path)

    # To force reload: chatbot.load_pdf_knowledge(pdf_path, force_reload=True)

    print("\n✅ RAG Chatbot is ready! Type 'quit' to exit.\n")
    print("💡 Ask questions about the Drugs and Cosmetics Act 1940!")
    print("\n📝 Sample queries you can try:")

    sample_queries = [
        # General Definitions & Scope
        "What is the definition of 'drug' under the Drugs and Cosmetics Act, 1940?",
        "How does the Act define 'cosmetic'?",
        "What is the meaning of 'standard quality' for drugs and cosmetics?",
        "What is the scope of the Drugs and Cosmetics Act, 1940?",

        # Authorities & Committees
        "What is the role of the Drugs Technical Advisory Board?",
        "Who are the members of the Central Drugs Laboratory?",
        "What is the function of the Drugs Consultative Committee?",

        # Import, Manufacture, Sale, and Distribution
        "What are the standards for importing drugs into India?",
        "What are the conditions under which the manufacture of drugs is prohibited?",
        "Is a license required for the sale of drugs? What are the conditions?",
        "What records must be maintained by a licensed drug manufacturer?",

        # Penalties & Offences
        "What is the penalty for manufacturing or selling adulterated drugs?",
        "What is the punishment for the import of spurious drugs?",
        "What are the consequences for repeated offences under the Act?",
        "What is the penalty for non-disclosure of the name of the drug manufacturer?",

        # Special Provisions
        "What are the special provisions for Ayurvedic, Siddha, or Unani drugs?",
        "Does the Act apply to government departments manufacturing drugs?",
        "What are the powers of inspectors under the Act?",

        # Procedures & Enforcement
        "What is the procedure for the analysis of drug samples by Government Analysts?",
        "How can a consumer get a drug tested under the Act?",
        "What are the powers of the Central Government to prohibit the manufacture or sale of drugs?",

        # Miscellaneous
        "What is the process for amending the schedules under the Act?",
        "How are offences by companies handled under the Act?",
        "What are the protections for actions taken in good faith under the Act?"
    ]

    for i, query in enumerate(sample_queries[:10], 1):  # Show first 10 queries
        print(f"  {i}. {query}")

    print(f"\n   ... and {len(sample_queries) - 10} more queries available!")
    print("\n🎯 Type 'samples' to see all sample queries")
    print("="*60)

    # Chat loop
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("👋 Goodbye!")
                break

            if user_input.lower() == 'samples':
                print("\n📝 All Sample Queries:\n")
                for i, query in enumerate(sample_queries, 1):
                    print(f"{i:2d}. {query}")
                print("\n" + "="*60)
                continue

            if not user_input:
                continue

            print("\n🤖 Bot:")
            chatbot.print_chat(user_input)
            print("\n" + "="*60)

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
