from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration for the whole app.

    Every field below maps to an environment variable of the same name
    (case-insensitive). Pydantic reads values from .env automatically.

    If a required value is missing, the app crashes at startup with a
    clear error message — not somewhere random during a request.
    """

    # --- Qdrant (vector DB) ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6Y2Q3MWUyMTQtZmZmYy00ZWNmLWIwYzktYzYwZTY3OGRlZGZmIn0.EPND10-QhqPOVsEJlW_CRnSTk1kdteortBqBM7TP2O8"

    # --- LLM routing via LiteLLM ---
    # These are model strings LiteLLM understands.
    # "groq/llama3-8b-8192" means: use Groq's API, model llama3-8b-8192
    embedding_model: str = "ollama/nomic-embed-text"
    extraction_model: str = "groq/llama3-8b-8192"

    # --- API keys ---
    # No default — if this is missing, Settings() raises an error.
    # That's intentional: better to fail at startup than mid-request.
    groq_api_key: str
    api_key: str = "dev-key-change-in-production"

    # --- Supabase (not used yet, but declared for later phases) ---
    supabase_url: str = ""
    supabase_key: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignore any .env vars not defined above
    )


# Single shared instance — every other file imports THIS object,
# never creates its own Settings().
settings = Settings()