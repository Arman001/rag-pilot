# config/settings.py
import os
from pathlib import Path
from typing import List, Tuple, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, FieldValidationInfo, SecretStr
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Gemini Configuration
    GEMINI_MODEL: str = Field(
        default="gemini-2.0-flash",
        description="Gemini model version",
        env="GEMINI_MODEL"
    )
    GOOGLE_API_KEY: SecretStr = Field(
        default=os.getenv("GOOGLE_API_KEY"),
        description="Google AI Studio API Key",
        env="GOOGLE_API_KEY"
    )
    
    # Directory Configurations
    BASE_URL: str = Field(
        default="https://python.langchain.com/docs/introduction/",
        description="Base URL for documentation scraping"
    )
    RAW_DATA_DIR: str = Field(
        default="data/langchain/raw",
        env="RAW_DATA_PATH"
    )
    PROCESSED_DATA_DIR: str = Field(
        default="data/langchain/processed", 
        env="PROCESSED_DATA_PATH"
    )
    VECTORSTORE_DIR: str = Field(
        default="vectorstore",
        env="VECTORSTORE_PATH"
    )
    CACHE_DIR: str = Field(
        default="cache",
        env="CACHE_PATH"
    )
    
    # Embedding Model
    EMBEDDING_MODEL: str = Field(
        default="BAAI/bge-small-en",
        description="FastEmbed model name",
        env="EMBEDDING_MODEL"
    )
    
    # Retrieval Parameters
    MIN_SIMILARITY_SCORE: float = Field(
        default=0.15,  # Lowered for better recall
        ge=0.0,
        le=1.0,
        description="Minimum relevance score (0.0-1.0)",
        env="MIN_SIMILARITY"
    )
    
    RETRIEVAL_TEST_CASES: List[Tuple[str, int, Optional[List[str]]]] = Field(
        default=[
            ("What is LangChain?", 1, ["introduction_0"]),
            ("How to install LangChain?", 1, ["introduction_2"]),
            ("Explain LangGraph", 2, ["introduction_0", "introduction_5"])
        ],
        description="(query, min_results, expected_chunk_ids)"
    )
    
    # Performance
    EMBEDDING_THREADS: int = Field(
        default=min(4, (os.cpu_count() or 1)),
        ge=1,
        le=16,
        description="Threads for parallel embedding",
        env="EMBEDDING_THREADS"
    )
    
    # Request Handling
    REQUEST_DELAY: float = Field(
        default=1.0,
        description="Delay between web requests (seconds)"
    )
    
    # Configuration
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        case_sensitive=False,
        frozen=False,
        extra="ignore"
    )
    
    @field_validator('*', mode='before')
    def ensure_paths(cls, v: str, info: FieldValidationInfo) -> str:
        """Ensure directory paths exist and return absolute paths"""
        if info.field_name.endswith(('_DIR', '_PATH')):
            path = Path(v)
            path.mkdir(parents=True, exist_ok=True)
            return str(path.absolute())
        return v
    
    @field_validator('GOOGLE_API_KEY')
    def validate_api_key(cls, v: SecretStr) -> SecretStr:
        if not v.get_secret_value():
            raise ValueError("Google API key is required")
        return v

# Singleton instance
settings = Settings()

if __name__ == "__main__":
    from rich import print
    print("[bold]Current Settings:[/bold]")
    print(settings.model_dump(exclude={"GOOGLE_API_KEY"}))
    print(f"Gemini configured: {'✅' if settings.GOOGLE_API_KEY.get_secret_value() else '❌'}")