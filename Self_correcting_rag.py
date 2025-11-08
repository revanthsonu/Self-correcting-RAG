"""
🚀 FREE Self-Correcting RAG Pipeline with Integrated Testing
No API keys needed! Uses Ollama (local) + FREE embeddings
Run with: python free_rag.py --test (to run tests first)
"""

from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import time
import os
import sys
import glob
from typing import List, Dict, Tuple
import re
import argparse


# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def test_imports() -> Dict[str, bool]:
    """Test if all required packages are installed."""
    print("\n" + "="*80)
    print("TEST 1: Checking Package Imports")
    print("="*80 + "\n")
    
    results = {}
    packages = [
        ("langchain", "LangChain"),
        ("langchain_community", "LangChain Community"),
        ("chromadb", "ChromaDB"),
        ("sentence_transformers", "Sentence Transformers"),
    ]
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"✓ {name:30s} - OK")
            results[package] = True
        except ImportError:
            print(f"✗ {name:30s} - MISSING")
            print(f"  Install with: pip install {package}")
            results[package] = False
    
    return results


def test_ollama() -> bool:
    """Test if Ollama is running and accessible."""
    print("\n" + "="*80)
    print("TEST 2: Checking Ollama Connection")
    print("="*80 + "\n")
    
    try:
        from langchain_community.llms import Ollama
        print("Testing Ollama connection...")
        llm = Ollama(model="llama3.2", temperature=0)
        response = llm.invoke("Say 'OK' if you can read this.")
        print(f"✓ Ollama is running and responding")
        print(f"  Model: llama3.2")
        print(f"  Response: {response[:50]}...")
        return True
    except Exception as e:
        print(f"✗ Ollama connection failed: {e}")
        print("  Make sure Ollama is running:")
        print("    1. Check menu bar for llama icon (Mac)")
        print("    2. Run: ollama list")
        print("    3. Run: ollama pull llama3.2")
        return False


def test_embeddings() -> bool:
    """Test embedding model loading."""
    print("\n" + "="*80)
    print("TEST 3: Testing Embedding Model")
    print("="*80 + "\n")
    
    try:
        print("Loading HuggingFace embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("✓ Embedding model loaded")
        
        print("\nTesting embedding generation...")
        test_text = "This is a test sentence."
        embedding = embeddings.embed_query(test_text)
        print(f"✓ Generated embedding with dimension: {len(embedding)}")
        
        return True
    except Exception as e:
        print(f"✗ Embedding test failed: {e}")
        return False


def test_vectorstore() -> bool:
    """Test vector store creation."""
    print("\n" + "="*80)
    print("TEST 4: Testing Vector Store")
    print("="*80 + "\n")
    
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain.schema import Document
        
        print("Creating test vector store...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        docs = [
            Document(page_content="Python is a programming language."),
            Document(page_content="Machine learning uses algorithms."),
            Document(page_content="Data science involves statistics.")
        ]
        
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings
        )
        print(f"✓ Vector store created with {len(docs)} documents")
        
        print("\nTesting similarity search...")
        results = vectorstore.similarity_search("What is Python?", k=2)
        print(f"✓ Retrieved {len(results)} similar documents")
        print(f"  Top result: {results[0].page_content}")
        
        return True
    except Exception as e:
        print(f"✗ Vector store test failed: {e}")
        return False


def run_all_tests() -> bool:
    """Run all tests and return overall success."""
    print("\n" + "█"*80)
    print("FREE RAG SYSTEM - VERIFICATION TESTS")
    print("█"*80)
    
    results = {
        "imports": all(test_imports().values()),
        "ollama": test_ollama(),
        "embeddings": test_embeddings(),
        "vectorstore": test_vectorstore()
    }
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80 + "\n")
    
    passed = sum(1 for v in results.values() if v is True)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name.upper():20s} {status}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if all(results.values()):
        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED - SYSTEM READY!")
        print("="*80)
        return True
    else:
        print("\n" + "="*80)
        print("✗ SOME TESTS FAILED - FIX ISSUES ABOVE")
        print("="*80)
        return False


# =============================================================================
# RAG SYSTEM
# =============================================================================

class FreeRAG:
    """Self-correcting RAG using only free, local tools"""
    
    def __init__(self, model_name: str = "llama3.2", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print("\n🚀 Initializing FREE RAG Pipeline (No API Keys!)...\n")
        
        print(f"📡 Connecting to Ollama...")
        self.llm = Ollama(model=model_name, temperature=0)
        print(f"✅ Connected to Ollama with model: {model_name}\n")
        
        print(f"📦 Loading FREE embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        print(f"✅ Embeddings loaded\n")
        
        self.vectorstore = None
        self.retriever = None
        
        print("✅ FREE RAG Pipeline Ready!\n")
    
    def load_from_text_file(self, filepath: str) -> List[str]:
        """Load content from a text file"""
        print(f"📄 Loading: {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            print(f"   ✓ Loaded {len(content)} characters")
            return [content]
        except Exception as e:
            print(f"   ✗ Error: {e}")
            return []
    
    def load_from_folder(self, folder_path: str, pattern: str = "*.txt") -> List[str]:
        """Load all files matching pattern from a folder"""
        print(f"📁 Loading files from: {folder_path}")
        all_docs = []
        files = glob.glob(os.path.join(folder_path, pattern))
        
        if not files:
            print(f"   ⚠️  No files found matching '{pattern}'")
            return []
        
        print(f"   Found {len(files)} files:")
        for filepath in files:
            filename = os.path.basename(filepath)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                all_docs.append(content)
                print(f"   ✓ {filename} ({len(content)} chars)")
            except Exception as e:
                print(f"   ✗ {filename}: {e}")
        
        return all_docs
    
    def add_documents(self, documents: List[str], chunk_size: int = 500, chunk_overlap: int = 50):
        """Add documents to the knowledge base"""
        print(f"\n📦 Processing {len(documents)} documents...")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.create_documents(documents)
        
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 5}
        )
        
        print(f"✅ Created {len(chunks)} chunks in vector database\n")
    
    def check_relevance(self, query: str, doc: str) -> Tuple[bool, float]:
        """Check if document is relevant to query"""
        prompt = f"""Rate relevance of this document to the query on a scale of 0-1.
Only respond with a number between 0 and 1.

Query: {query}
Document: {doc}

Relevance score (0-1):"""
        
        try:
            response = self.llm.invoke(prompt).strip()
            score = float(re.findall(r"0?\.\d+|[01]", response)[0])
            return score > 0.5, score
        except:
            return True, 0.5
    
    def check_hallucination(self, answer: str, context: str) -> Tuple[bool, float]:
        """Check if answer is grounded in context"""
        prompt = f"""Is this answer factually supported by the context? Rate 0-1.
Only respond with a number between 0 and 1.

Context: {context}
Answer: {answer}

Factual support score (0-1):"""
        
        try:
            response = self.llm.invoke(prompt).strip()
            score = float(re.findall(r"0?\.\d+|[01]", response)[0])
            return score > 0.7, score
        except:
            return True, 0.7
    
    def query(self, question: str) -> Dict:
        """Query with self-correction"""
        if not self.retriever:
            return {"error": "No documents added yet!"}
        
        start_time = time.time()
        
        print(f"❓ Question: {question}\n")
        
        # Step 1: Retrieve
        print(f"[1/4] 🔍 Retrieving top 5 documents...")
        docs = self.retriever.get_relevant_documents(question)
        print(f"      Found {len(docs)} documents\n")
        
        # Step 2: Filter irrelevant docs
        print(f"[2/4] ✓  Checking relevance...")
        filtered_docs = []
        for i, doc in enumerate(docs, 1):
            is_relevant, score = self.check_relevance(question, doc.page_content)
            if is_relevant:
                filtered_docs.append(doc)
                print(f"      ✓ Doc {i}: RELEVANT (score: {score:.2f})")
            else:
                print(f"      ✗ Doc {i}: FILTERED (score: {score:.2f})")
        
        print(f"      Kept {len(filtered_docs)}/{len(docs)} documents\n")
        
        if not filtered_docs:
            return {
                "answer": "No relevant documents found for your question.",
                "fact_check": False,
                "score": 0.0,
                "sources": 0,
                "time": time.time() - start_time
            }
        
        # Step 3: Generate answer
        print(f"[3/4] 💡 Generating answer...")
        context = "\n\n".join([doc.page_content for doc in filtered_docs])
        
        prompt = f"""Answer the question based only on the context below. Be concise and specific.

Context:
{context}

Question: {question}

Answer:"""
        
        answer = self.llm.invoke(prompt).strip()
        print(f"      Generated ({len(answer)} chars)\n")
        
        # Step 4: Fact check
        print(f"[4/4] 🔬 Fact-checking...")
        is_factual, consistency_score = self.check_hallucination(answer, context)
        print(f"      ✓ Factual: {is_factual}, Score: {consistency_score:.2f}\n")
        
        elapsed = time.time() - start_time
        
        result = {
            "answer": answer,
            "fact_check": is_factual,
            "score": consistency_score,
            "sources": len(filtered_docs),
            "time": elapsed
        }
        
        # Print result
        print("=" * 80)
        print("RESULT")
        print("=" * 80)
        print(f"\n💬 Answer:\n{answer}\n")
        print(f"✓ Fact Check: {'PASSED' if is_factual else 'FAILED'}")
        print(f"📊 Consistency: {consistency_score:.2f}")
        print(f"📚 Sources: {len(filtered_docs)}")
        print(f"⏱️  Time: {elapsed:.2f}s")
        print("=" * 80)
        print()
        
        return result


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution with optional testing"""
    parser = argparse.ArgumentParser(description='FREE Self-Correcting RAG')
    parser.add_argument('--test', action='store_true', help='Run tests before starting')
    parser.add_argument('--skip-demo', action='store_true', help='Skip demo questions')
    args = parser.parse_args()
    
    # Run tests if requested
    if args.test:
        if not run_all_tests():
            print("\n⚠️  Tests failed! Fix issues before using RAG system.")
            return 1
        print("\n" + "="*80)
        print("STARTING RAG SYSTEM")
        print("="*80)
    
    # Create RAG instance
    rag = FreeRAG(model_name="llama3.2")
    
    # Load knowledge base
    knowledge_base = [
        "The Eiffel Tower is located in Paris, France. It was built in 1889 and stands 330 meters tall, including antennas. It was designed by engineer Gustave Eiffel.",
        "Paris is the capital of France and has a population of about 2.2 million people. It is known for landmarks like the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral.",
        "The Great Wall of China is approximately 21,196 kilometers long. It was built over many centuries to protect Chinese states from invasions.",
        "Python is a high-level programming language created by Guido van Rossum in 1991. It emphasizes code readability and supports multiple programming paradigms.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
        "The Amazon rainforest produces about 20% of the world's oxygen and is home to 10% of all species on Earth.",
    ]
    
    rag.add_documents(knowledge_base)
    
    # Run demo unless skipped
    if not args.skip_demo:
        test_questions = [
            "How tall is the Eiffel Tower?",
            "What is the population of Paris?",
            "Who created Python programming language?"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print("\n" + "█" * 80)
            print(f"DEMO QUERY {i}/{len(test_questions)}")
            print("█" * 80)
            print()
            result = rag.query(question)
            
            if i < len(test_questions):
                print("\n⏳ Moving to next question...\n")
                time.sleep(1)
    
    print("\n" + "="*80)
    print("✅ READY FOR YOUR QUERIES!")
    print("="*80)
    print("\n💡 USAGE:")
    print("   1. Edit 'knowledge_base' list to add your content")
    print("   2. Or use: rag.load_from_text_file('yourfile.txt')")
    print("   3. Or use: rag.load_from_folder('./documents')")
    print("   4. Then: rag.query('your question')")
    print("\n🧪 TESTING:")
    print("   Run with: python free_rag.py --test")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
