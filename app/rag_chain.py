#!/usr/bin/env python3
"""Enhanced RAG Pipeline with optimized retrieval and response generation"""

from pathlib import Path
import os
import google.generativeai as genai
import numpy as np
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from typing import List, Optional, Union
from pydantic import SecretStr
from config.settings import settings


class RAGPipeline:
    def __init__(self):
        """Initialize RAG pipeline with comprehensive validation"""
        try:
            self._validate_settings()
            self._configure_gemini()
            self._initialize_vectorstore()
            self._initialize_retriever()
            self._initialize_llm()
            self._setup_processing_chain()
            
            # Non-critical system checks
            try:
                self._run_system_checks()
            except Exception as e:
                print(f"⚠️ System check warning: {str(e)}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize RAGPipeline: {str(e)}")

    def _validate_settings(self):
        """Validate required configuration"""
        required_settings = {
            'EMBEDDING_MODEL': (str,),
            'GOOGLE_API_KEY': (str, SecretStr),
            'VECTORSTORE_DIR': (str,),
            'GEMINI_MODEL': (str,)
        }
        
        for setting, accepted_types in required_settings.items():
            value = getattr(settings, setting, None)
            
            if isinstance(value, SecretStr):
                if not value.get_secret_value():
                    raise ValueError(f"Empty secret value for {setting}")
            elif not value or not isinstance(value, accepted_types):
                raise ValueError(
                    f"Invalid setting: {setting}. "
                    f"Expected {accepted_types}, got {type(value)}"
                )

    def _configure_gemini(self):
        """Configure Gemini API"""
        genai.configure(api_key=settings.GOOGLE_API_KEY.get_secret_value())

    def _initialize_vectorstore(self):
        """Load and validate vector store"""
        try:
            self.vectorstore = FAISS.load_local(
            settings.VECTORSTORE_DIR,
            FastEmbedEmbeddings(
                model_name=settings.EMBEDDING_MODEL,
                cache_dir=str(Path(settings.CACHE_DIR).absolute()),
                threads=min(4, (os.cpu_count() or 1)),
                show_progress_bar=True,
            ),
            allow_dangerous_deserialization=True,
        )
            print(f"✅ Vectorstore loaded with {self.vectorstore.index.ntotal} documents")
            
            # Verify embedding dimensions
            sample_embed = FastEmbedEmbeddings(
                model_name=settings.EMBEDDING_MODEL
            ).embed_query("test")
            if len(sample_embed) != self.vectorstore.index.d:
                raise ValueError(
                    f"Embedding dimension mismatch. "
                    f"Model: {len(sample_embed)}, Vectorstore: {self.vectorstore.index.d}"
                )
                
        except Exception as e:
            raise RuntimeError(f"Vectorstore initialization failed: {str(e)}")

    def _initialize_retriever(self):
        """Configure document retriever with optimized settings"""
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",  # Use Maximal Marginal Relevance
            search_kwargs={
                "k": 5,
                "lambda_mult": 0.7,  # Balances diversity vs relevance
                "score_threshold": 0.15  # Lower threshold
            }
        )

    def _initialize_llm(self):
        """Initialize language model"""
        self.llm = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            temperature=0.3,
            convert_system_message_to_human=True,
        )

    def _run_system_checks(self):
        """Execute diagnostic checks"""
        # Test retrieval with sample query
        test_query = "What is LangChain?"
        test_docs = self.vectorstore.similarity_search(test_query, k=3)
        print(f"\n🔍 System check: Found {len(test_docs)} documents for '{test_query}'")
        
        # Check embedding quality
        good_query = "LangChain framework"
        emb = FastEmbedEmbeddings().embed_query(good_query)
        scores = np.array([
            np.dot(emb, self.vectorstore.index.reconstruct(i))
            for i in range(min(10, self.vectorstore.index.ntotal))
        ])
        print(f"Average similarity for known good query: {np.mean(scores):.3f}")

    def _setup_processing_chain(self):
        """Configure the RAG processing chain"""
        self.prompt = ChatPromptTemplate.from_template("""
        You're a technical assistant for LangChain documentation. 
        Answer the question using ONLY the provided context.

        Important Rules:
        1. Be precise and include key technical terms
        2. Always mention: LLMs (Large Language Models) when relevant
        3. Format code with ```triple backticks```
        4. If unsure, say "I couldn't find relevant information"

        Context: {context}
        Question: {question}

        Answer:
        """)
        
        self.chain = (
            {"context": self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_docs(self, docs: List[Document]) -> str:
        """Format documents for LLM input"""
        return "\n\n---\n\n".join(
            f"📄 Document {i+1} (ID: {doc.metadata.get('chunk_id', 'N/A')}:\n{doc.page_content}"
            for i, doc in enumerate(docs)
            if isinstance(doc, Document)
        )

    def query(self, question: str) -> str:
        """Run full RAG pipeline on input question and return LLM response"""
        print("query is called with question:", question)
        try:
            # Search vectorstore for top 5 most similar chunks
            docs_with_scores = self.vectorstore.similarity_search_with_score(question, k=5)

            print(f"\n🔎 Retrieved {len(docs_with_scores)} chunks for: '{question}'")
            for i, (doc, score) in enumerate(docs_with_scores):
                print(f"\n📌 Chunk {i+1} (Score: {score:.3f})")
                print(f"🆔 ID: {doc.metadata.get('chunk_id', 'N/A')}")
                print(f"📝 Content:\n{doc.page_content[:200]}...")

            docs = [doc for doc, _ in docs_with_scores]
            formatted_context = self._format_docs(docs)

            # Pass formatted context and question to the chain
            result = self.chain.invoke({
                "context": formatted_context,
                "question": question
            })

            return result

        except Exception as e:
            return f"⚠️ Error processing query: {str(e)}"

# create a singleton pipeline instance
rag_pipeline = RAGPipeline()

# Expose the chain for external use
qa_chain = rag_pipeline.chain