from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "medgemma"
    OLLAMA_TIMEOUT: int = 120

    # Database
    DATABASE_URL: str = "sqlite:///./data/medgemma_agent.db"

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./data/chroma"

    # Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_CACHE_DIR: str = "./data/models"

    # API Keys (comma-separated per role)
    API_KEYS_ADMIN: str = ""
    API_KEYS_CLINICIAN: str = ""
    API_KEYS_READONLY: str = ""

    # Encryption — generate with:
    # python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    ENCRYPTION_KEY: str = ""

    # Compliance
    AUDIT_RETENTION_YEARS: int = 6

    # LLM behaviour
    LLM_FALLBACK_ON_TIMEOUT: bool = True
    LLM_CONFIDENCE_THRESHOLD: float = 0.6

    # Development mode — disables API key auth for local testing only
    # NEVER set to true in production
    DEV_MODE: bool = False

    model_config = {"env_file": ".env", "extra": "ignore"}

    def admin_keys(self) -> set[str]:
        return {k.strip() for k in self.API_KEYS_ADMIN.split(",") if k.strip()}

    def clinician_keys(self) -> set[str]:
        return {k.strip() for k in self.API_KEYS_CLINICIAN.split(",") if k.strip()}

    def readonly_keys(self) -> set[str]:
        return {k.strip() for k in self.API_KEYS_READONLY.split(",") if k.strip()}

    def all_valid_keys(self) -> dict[str, str]:
        """Returns mapping of api_key → role."""
        keys: dict[str, str] = {}
        for k in self.admin_keys():
            keys[k] = "admin"
        for k in self.clinician_keys():
            keys[k] = "clinician"
        for k in self.readonly_keys():
            keys[k] = "readonly"
        return keys


@lru_cache
def get_settings() -> Settings:
    return Settings()
