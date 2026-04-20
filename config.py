from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    pinecone_api_key: str
    gemini_api_key: str                         
    pinecone_index_name: str = "medical-assistant"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 3
    hf_token: Optional[str] = None  

    class Config:
        env_file = ".env"
        extra = "ignore"
        env_file_encoding = "utf-8" 

settings = Settings()