from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # API keys
    pinecone_api_key: str
    gemini_api_key: str
    hf_token: Optional[str] = None
    cohere_api_key: Optional[str] = None

    # Pinecone
    pinecone_index_name: str = "medical-assistant"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # App
    secret_key: str = None
    database_url: str = None

    class Config:
        env_file = ".env"
        extra = "ignore"
        env_file_encoding = "utf-8"


settings = Settings()