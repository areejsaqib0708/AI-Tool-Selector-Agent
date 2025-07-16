import os
import re
import chromadb
import google.generativeai as genai
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from API import api

api_key=api()

# Configuration
TEXT_DIR = "C:/Users/Public/Internship/admin_data"
os.makedirs(TEXT_DIR, exist_ok=True)
FILE_PATH = os.path.join(TEXT_DIR, "admin_text.txt")
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-1.5-flash"

# Create directories if not exists
os.makedirs(TEXT_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

class RAGSystem:
    TEXT_DIR = "C:/Users/Public/Internship/admin_data"
    def __init__(self, gemini_api_key):
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            embedding_function=self.ef
        )

        # Initialize Gemini
        genai.configure(api_key=gemini_api_key)
        self.gemini = genai.GenerativeModel(GEMINI_MODEL)
        self.vectorized_docs = set()  # Track processed documents

    def _clean_text(self, text):
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace
        text = re.sub(r'\u202f|\u200b', ' ', text)  # Remove special spaces
        return text.strip()

    def _chunk_text(self, text, chunk_size=1000, overlap=100):
        """Split text into semantic chunks with overlap"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sent_length = len(sentence.split())
            if current_length + sent_length > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Keep overlap for context
                current_chunk = current_chunk[-int(len(current_chunk) * overlap / chunk_size):]
                current_length = sum(len(word) for word in current_chunk)

            current_chunk.append(sentence)
            current_length += sent_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def process_document(self, file_path):
        """Process text document and store embeddings in ChromaDB"""
        # Skip if already processed
        doc_name = os.path.basename(file_path)
        if doc_name in self.vectorized_docs:
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            text = self._clean_text(f.read())

        chunks = self._chunk_text(text)
        if not chunks:
            return

        # Generate embeddings
        embeddings = self.embedding_model.encode(chunks)

        # Store in ChromaDB
        ids = [f"{doc_name}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": doc_name} for _ in chunks]

        self.collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        self.vectorized_docs.add(doc_name)

    def query(self, question, top_k=5):
        """Retrieve relevant context and generate answer"""
        # Embed question
        query_embedding = self.embedding_model.encode([question]).tolist()[0]

        # Retrieve relevant chunks
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas"]
        )

        # Safely combine context with sources
        context_chunks = []
        for i, doc in enumerate(results['documents'][0]):
            # Handle cases where metadata is None or missing
            meta = results['metadatas'][0][i] if (results['metadatas'] and i < len(results['metadatas'][0])) else {}
            source = meta.get('source', 'Unknown document') if isinstance(meta, dict) else str(meta)
            context_chunks.append(f"[Source: {source}]\n{doc}")

        context = "\n\n---\n\n".join(context_chunks)

        # Generate response
        prompt = f"""Answer the following question in a full sentence using ONLY the context below.
        Do not skip any relevant information from the context.
        If the context is not useful, answer based on your own knowledge.
        If you are not sure, say "I couldn't find relevant information."
        Context:
        {context}
        Question: {question}
        Answer:"""

        try:
            response = self.gemini.generate_content(prompt)
            return response.text, context
        except Exception as e:
            return f"Error generating response: {str(e)}", context
if __name__ == "__main__":
    GEMINI_API_KEY = api_key
    rag = RAGSystem(GEMINI_API_KEY)

    # Process documents from text directory
    for filename in os.listdir(TEXT_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(TEXT_DIR, filename)
            rag.process_document(file_path)
            print(f"Processed: {filename}")

    # Interactive Q&A
    print("\nRAG System Ready. Type 'exit' to quit")
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'exit':
            break
        answer, context = rag.query(question)
        print("\n" + "=" * 50)
        print(answer)
        print("=" * 50)
        # Optional: show context sources
        show_context = input("\nShow sources? (y/n): ")
        if show_context.lower() == 'y':
            print("\nCONTEXT SOURCES:")
            print(context)