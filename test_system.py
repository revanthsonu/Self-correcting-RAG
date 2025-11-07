"""
Test Script to Verify Self-Correcting RAG Pipeline
Run this to check if everything is working correctly
"""

import os
import sys
from typing import Dict, List

def test_imports() -> Dict[str, bool]:
    """Test if all required packages are installed."""
    print("\n" + "="*80)
    print("TEST 1: Checking Package Imports")
    print("="*80 + "\n")
    
    results = {}
    
    packages = [
        ("langchain", "LangChain"),
        ("langchain_community", "LangChain Community"),
        ("langchain_openai", "LangChain OpenAI"),
        ("chromadb", "ChromaDB"),
        ("sentence_transformers", "Sentence Transformers"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
    ]
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"✓ {name:30s} - OK")
            results[package] = True
        except ImportError as e:
            print(f"✗ {name:30s} - MISSING")
            print(f"  Install with: pip install {package}")
            results[package] = False
    
    return results


def test_environment() -> bool:
    """Test environment setup."""
    print("\n" + "="*80)
    print("TEST 2: Checking Environment Variables")
    print("="*80 + "\n")
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key:
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        print(f"✓ OPENAI_API_KEY found: {masked_key}")
        return True
    else:
        print("✗ OPENAI_API_KEY not set")
        print("  Set with: export OPENAI_API_KEY='your-key'")
        return False


def test_basic_functionality():
    """Test basic RAG functionality without API calls."""
    print("\n" + "="*80)
    print("TEST 3: Basic Functionality (No API Calls)")
    print("="*80 + "\n")
    
    try:
        # Test document splitting
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document
        
        print("Testing document splitting...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        docs = [Document(page_content="This is a test document. " * 20)]
        splits = text_splitter.split_documents(docs)
        print(f"✓ Split 1 document into {len(splits)} chunks")
        
        # Test embeddings initialization
        print("\nTesting embeddings initialization...")
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("✓ Embeddings model loaded")
        
        # Test embedding generation
        print("\nTesting embedding generation...")
        test_text = "This is a test sentence."
        embedding = embeddings.embed_query(test_text)
        print(f"✓ Generated embedding with dimension: {len(embedding)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_vectorstore():
    """Test vector store creation."""
    print("\n" + "="*80)
    print("TEST 4: Vector Store Creation")
    print("="*80 + "\n")
    
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain.schema import Document
        
        print("Creating test vector store...")
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Create test documents
        docs = [
            Document(page_content="Python is a programming language."),
            Document(page_content="Machine learning uses algorithms."),
            Document(page_content="Data science involves statistics.")
        ]
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings
        )
        print(f"✓ Vector store created with {len(docs)} documents")
        
        # Test search
        print("\nTesting similarity search...")
        results = vectorstore.similarity_search("What is Python?", k=2)
        print(f"✓ Retrieved {len(results)} similar documents")
        print(f"  Top result: {results[0].page_content[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_structure():
    """Test RAG class structure without API calls."""
    print("\n" + "="*80)
    print("TEST 5: RAG Class Structure")
    print("="*80 + "\n")
    
    try:
        # Check if API key exists for full test
        api_key = os.getenv("OPENAI_API_KEY", "sk-test-key-for-initialization")
        
        print("Importing RAG classes...")
        from self_correcting_rag import (
            SelfCorrectingRAG,
            RelevanceAgent,
            GeneratorAgent,
            FactCheckAgent,
            RelevanceScore,
            FactCheckResult,
            RAGResult
        )
        print("✓ All classes imported successfully")
        
        # Test dataclass structures
        print("\nTesting dataclass structures...")
        
        relevance = RelevanceScore(
            document_id="test",
            is_relevant=True,
            score=0.8,
            reasoning="Test reasoning"
        )
        print(f"✓ RelevanceScore: {relevance.score}")
        
        fact_check = FactCheckResult(
            is_factual=True,
            consistency_score=0.9,
            issues=[],
            verified_claims=["Test claim"]
        )
        print(f"✓ FactCheckResult: {fact_check.consistency_score}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_api():
    """Test full pipeline with API (optional)."""
    print("\n" + "="*80)
    print("TEST 6: Full Pipeline with API (Optional)")
    print("="*80 + "\n")
    
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("⊘ Skipped - No API key configured")
        print("  Set OPENAI_API_KEY to run this test")
        return None
    
    try:
        from self_correcting_rag import SelfCorrectingRAG
        
        print("Initializing RAG pipeline...")
        rag = SelfCorrectingRAG(
            openai_api_key=api_key,
            llm_model="gpt-3.5-turbo",
            temperature=0.0,
            relevance_threshold=0.6
        )
        print("✓ Pipeline initialized")
        
        print("\nAdding test documents...")
        test_docs = [
            "The capital of France is Paris.",
            "Python was created by Guido van Rossum.",
            "Machine learning is a subset of AI."
        ]
        rag.add_documents(test_docs)
        print(f"✓ Added {len(test_docs)} documents")
        
        print("\nRunning test query...")
        result = rag.query("What is the capital of France?", top_k=3)
        print(f"✓ Query executed in {result.metadata['processing_time']:.2f}s")
        print(f"  Answer: {result.answer[:100]}...")
        print(f"  Fact Check: {result.fact_check.is_factual}")
        print(f"  Consistency: {result.fact_check.consistency_score:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "█"*80)
    print("SELF-CORRECTING RAG - INSTALLATION VERIFICATION")
    print("█"*80)
    
    # Run tests
    results = {
        "imports": test_imports(),
        "environment": test_environment(),
        "functionality": test_basic_functionality(),
        "vectorstore": test_vectorstore(),
        "structure": test_rag_structure(),
        "api": test_with_api()
    }
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80 + "\n")
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    print(f"✓ Passed:  {passed}")
    print(f"✗ Failed:  {failed}")
    print(f"⊘ Skipped: {skipped}")
    
    if failed == 0 and passed >= 4:
        print("\n" + "="*80)
        print("✓ ALL CRITICAL TESTS PASSED - READY TO USE!")
        print("="*80)
        return 0
    elif failed > 0:
        print("\n" + "="*80)
        print("✗ SOME TESTS FAILED - FIX ISSUES ABOVE")
        print("="*80)
        return 1
    else:
        print("\n" + "="*80)
        print("⚠ PARTIAL SUCCESS - REVIEW WARNINGS")
        print("="*80)
        return 0



from free_rag import run_all_tests
import sys

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
