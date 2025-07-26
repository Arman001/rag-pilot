#!/usr/bin/env python3
"""Enhanced Embedding Pipeline with Comprehensive Testing"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import FastEmbedEmbeddings
from config.settings import settings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Test cases tailored to the introduction.md content
TEST_CASES = [
    {
        "name": "Framework Definition",
        "query": "What is LangChain?",
        "expected_chunk_id": "introduction_0",
        "min_score": 0.25,
        "filter": lambda m: True
    },
    {
        "name": "Core Components",
        "query": "What are the main components of LangChain?",
        "expected_chunk_id": ["introduction_4", "introduction_5"],
        "min_score": 0.2,
        "filter": lambda m: "Header 2" in m and m["Header 2"] == "Architecture"
    },
    {
        "name": "Installation Command",
        "query": "How to install LangChain for Google Gemini?",
        "expected_chunk_id": "introduction_2",
        "min_score": 0.15,
        "filter": lambda m: "pip install" in m.get("page_content", "").lower()
    },
    {
        "name": "Getting Started",
        "query": "Where can I find LangChain tutorials?",
        "expected_chunk_id": "introduction_6",
        "min_score": 0.2,
        "filter": lambda m: "tutorials" in m.get("page_content", "").lower()
    },
    {
        "name": "Integration Providers",
        "query": "Which LLM providers does LangChain support?",
        "expected_chunk_id": "introduction_2",
        "min_score": 0.15,
        "filter": lambda m: any(provider in m.get("page_content", "") 
                              for provider in ["OpenAI", "Anthropic"])
    },
    {
        "name": "LangGraph Purpose",
        "query": "What is LangGraph used for?",
        "expected_chunk_id": ["introduction_0", "introduction_5"],
        "min_score": 0.18,
        "filter": lambda m: "LangGraph" in m.get("page_content", "")
    },
    {
        "name": "Negative Test",
        "query": "How to cook pasta?",
        "expected_chunk_id": None,
        "min_score": 0.1,
        "filter": lambda m: True
    }
]

def load_chunks(chunk_path: str) -> List[Document]:
    """Load and validate processed chunks"""
    try:
        path = Path(chunk_path)
        if not path.exists():
            raise FileNotFoundError(f"Chunk file not found at {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Empty chunk file at {path}")

        with open(path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        if not isinstance(chunks_data, list):
            raise ValueError(f"Expected list of documents, got {type(chunks_data)}")

        return [Document(**data) for data in chunks_data]

    except Exception as e:
        logger.error(f"Failed to load chunks: {str(e)}", exc_info=True)
        raise

def build_vectorstore(chunks: List[Document]) -> FAISS:
    """Build vectorstore with performance monitoring"""
    try:
        logger.info(f"Initializing embeddings with model: {settings.EMBEDDING_MODEL}")

        embeddings = FastEmbedEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            cache_dir=str(Path(settings.CACHE_DIR).absolute()),
            threads=min(4, (os.cpu_count() or 1)),
            show_progress_bar=True,
        )

        logger.info(f"Building FAISS index for {len(chunks)} chunks...")
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings,
            distance_strategy="COSINE"
        )

        if vectorstore.index.ntotal != len(chunks):
            raise ValueError(
                f"Index count mismatch. Expected {len(chunks)}, "
                f"got {vectorstore.index.ntotal}"
            )

        return vectorstore

    except Exception as e:
        logger.error("Failed to build vectorstore", exc_info=True)
        raise

def run_retrieval_tests(vectorstore: FAISS) -> Dict[str, Any]:
    """Run comprehensive retrieval tests with validation"""
    results = {}
    
    for case in TEST_CASES:
        try:
            # Perform the search
            docs = vectorstore.similarity_search_with_score(
                case["query"],
                k=3,
                filter=case["filter"]
            )
            
            # Process results
            case_results = []
            found_expected = False
            
            for doc, score in docs:
                if score < case["min_score"]:
                    continue
                    
                chunk_id = doc.metadata.get("chunk_id", "")
                page_content = doc.page_content[:150] + "..." if doc.page_content else ""
                
                # Handle both single and multiple expected chunks
                expected = case["expected_chunk_id"]
                is_expected = False
                
                if expected is not None:
                    if isinstance(expected, list):
                        is_expected = chunk_id in expected
                    else:
                        is_expected = chunk_id == expected
                
                result = {
                    "chunk_id": chunk_id,
                    "score": float(score),
                    "content_preview": page_content,
                    "is_expected": is_expected
                }
                
                if is_expected:
                    found_expected = True
                
                case_results.append(result)
            
            # Store test outcome
            results[case["name"]] = {
                "query": case["query"],
                "passed": found_expected or case["expected_chunk_id"] is None,
                "results": case_results,
                "expected_chunk": case["expected_chunk_id"],
                "min_score": case["min_score"]
            }
            
        except Exception as e:
            results[case["name"]] = {
                "error": str(e),
                "query": case["query"]
            }
    
    return results

def main() -> bool:
    """Main pipeline with enhanced testing"""
    try:
        logger.info("Starting embedding pipeline")

        # 1. Load chunks
        chunk_file = Path(settings.PROCESSED_DATA_DIR) / "introduction_chunks.json"
        chunks = load_chunks(str(chunk_file))
        logger.info(f"Loaded {len(chunks)} chunks from {chunk_file}")

        # Validate expected chunks exist
        expected_chunks = set()
        for case in TEST_CASES:
            if case['expected_chunk_id'] is not None:
                if isinstance(case['expected_chunk_id'], list):
                    expected_chunks.update(case['expected_chunk_id'])
                else:
                    expected_chunks.add(case['expected_chunk_id'])
        
        existing_chunks = {doc.metadata.get('chunk_id', '') for doc in chunks}
        
        if missing := expected_chunks - existing_chunks:
            logger.warning(f"Missing expected chunks: {missing}")

        # 2. Build vectorstore
        vectorstore = build_vectorstore(chunks)

        # 3. Save artifacts
        save_path = Path(settings.VECTORSTORE_DIR)
        vectorstore.save_local(str(save_path))
        logger.info(f"Saved vectorstore to {save_path.absolute()}")

        # 4. Run enhanced tests
        test_results = run_retrieval_tests(vectorstore)
        
        # 5. Print results
        print("\n=== Test Results ===")
        for test_name, result in test_results.items():
            status = "PASSED" if result.get('passed', False) else "FAILED"
            print(f"\n{test_name} ({status}):")
            print(f"Query: {result['query']}")
            
            if 'error' in result:
                print(f"ERROR: {result['error']}")
                continue
                
            for i, res in enumerate(result['results'], 1):
                print(f"[Result {i}] Chunk {res['chunk_id']} (Score: {res['score']:.3f})")
                print(f"Match: {'✅' if res['is_expected'] else '❌'}")
                print(f"Preview: {res['content_preview']}")

        # Return success if all expected tests passed
        return all(r.get('passed', False) for r in test_results.values() 
                  if r.get('expected_chunk') is not None)

    except Exception as e:
        logger.critical(f"Pipeline failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)