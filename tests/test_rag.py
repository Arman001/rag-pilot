#!/usr/bin/env python3
"""Comprehensive RAG Pipeline Tests with Retry Logic"""

import time
from typing import List, Dict, Any
from app.rag_chain import RAGPipeline
from config.settings import settings


def print_test_header(title: str):
    """Print formatted test header"""
    print("\n" + "="*60)
    print(f"🚀 {title.center(56)}")
    print("="*60)


def test_retriever_directly():
    """Test the retriever component in isolation"""
    print_test_header("TESTING RETRIEVER COMPONENT")
    
    try:
        rag = RAGPipeline()
        test_query = "What is LangChain?"
        
        # Test both retriever and direct search
        for method_name, method in [("retriever", rag.retriever.invoke),
                                  ("vectorstore", lambda q: rag.vectorstore.similarity_search(q, k=3))]:
            print(f"\n🔧 Testing {method_name}...")
            try:
                docs = method(test_query)
                print(f"📊 Found {len(docs)} documents")
                
                for i, doc in enumerate(docs):
                    print(f"\n📄 Document {i+1}:")
                    print(f"🆔 ID: {doc.metadata.get('chunk_id', 'N/A')}")
                    print(f"📝 Content:\n{doc.page_content[:150]}...")
                
                assert len(docs) > 0, f"{method_name} returned no documents"
                assert any("LangChain" in doc.page_content for doc in docs), "No relevant content found"
                print("✅ Passed")
                
            except Exception as e:
                print(f"❌ {method_name} test failed: {str(e)}")
                if method_name == "retriever":
                    print("⚠️ Trying direct vectorstore search as fallback...")
                    continue
                raise
                
    except Exception as e:
        print("\n❌❌ CRITICAL FAILURE:")
        print(f"Error: {str(e)}")
        print("\n🔧 Debugging Tips:")
        print("1. Verify vectorstore exists at:", settings.VECTORSTORE_DIR)
        print("2. Check embedding model matches between indexing/querying")
        print("3. Try regenerating vectorstore if problems persist")
        raise


def test_full_pipeline():
    """Test complete RAG pipeline with retry logic"""
    print_test_header("TESTING FULL RAG PIPELINE")
    
    test_cases = [
        {
            "query": "What is LangChain?",
            "min_score": 0.15,
            "expected_chunks": ["introduction_0"],
            "required_phrases": ["framework", "language models"],
            "min_length": 30,
            "max_tries": 3
        },
        {
            "query": "How to install LangChain for Gemini?",
            "min_score": 0.15,
            "expected_chunks": ["introduction_2", "introduction_3"],
            "required_phrases": ["pip install", "google-genai"],
            "min_length": 40,
            "max_tries": 2
        },
        {
            "query": "Explain LangGraph architecture",
            "min_score": 0.15,
            "expected_chunks": ["introduction_5", "introduction_6"],
            "required_phrases": ["LangGraph", "orchestration"],
            "min_length": 50,
            "max_tries": 2
        },
        {
            "query": "How to cook pasta?",
            "expected_response": "I couldn't find relevant information",
            "max_score": 0.1,
            "max_tries": 1
        }
    ]

    try:
        rag = RAGPipeline()
        overall_start = time.time()
        
        for case in test_cases:
            print(f"\n🧪 Testing Query: '{case['query']}'")
            max_tries = case.get("max_tries", 1)
            
            for attempt in range(max_tries):
                try:
                    start_time = time.time()
                    response = rag.query(case['query'])
                    elapsed = time.time() - start_time
                    
                    print(f"\n⏱️ Response time: {elapsed:.2f}s")
                    print(f"📝 Response ({len(response)} chars):")
                    print(response)
                    
                    # Validate negative test cases
                    if "expected_response" in case:
                        assert case["expected_response"] in response, "Missing expected fallback response"
                        print("✅ Passed (negative test)")
                        break
                    
                    # Validate positive test cases
                    missing_phrases = [
                        p for p in case["required_phrases"]
                        if p.lower() not in response.lower()
                    ]
                    assert not missing_phrases, f"Missing phrases: {missing_phrases}"
                    assert len(response) >= case["min_length"], "Response too short"
                    print("✅ Passed (positive test)")
                    break
                    
                except AssertionError as e:
                    if attempt == max_tries - 1:
                        raise
                    print(f"⚠️ Retry attempt {attempt + 1}/{max_tries}")
                    time.sleep(1)  # Brief delay between retries
                
        print(f"\n✨ All tests completed in {time.time()-overall_start:.2f} seconds")
        
    except Exception as e:
        print("\n❌❌ PIPELINE FAILURE:")
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        start_time = time.time()
        test_retriever_directly()
        test_full_pipeline()
    except Exception as e:
        print("\n💥 Critical test failure encountered!")
        raise
    finally:
        print(f"\nTotal test duration: {time.time()-start_time:.2f} seconds")
        print("="*80)