from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Hugging Face Access Token
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")

    # MLflow Tracking
    mlflow_tracking_uri: str | None = Field(default=None, alias="MLFLOW_TRACKING_URI")
    mlflow_tracking_username: str | None = Field(default=None, alias="MLFLOW_TRACKING_USERNAME")
    mlflow_tracking_password: str | None = Field(default=None, alias="MLFLOW_TRACKING_PASSWORD")

    # Qdrant Vector DB Configuration
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_grpc_port: int = Field(default=6334, alias="QDRANT_GRPC_PORT")

    # Storage paths
    mind_data_dir: Path = Field(default=Path("./data"), alias="MIND_DATA_DIR")
    artifacts_dir: Path = Field(default=Path("./artifacts"), alias="ARTIFACTS_DIR")

    # Models & Stage parameters
    embedding_model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        alias="EMBEDDING_MODEL_NAME",
    )
    candidate_top_k: int = Field(default=100, alias="CANDIDATE_TOP_K")
    ranker_top_k: int = Field(default=10, alias="RANKER_TOP_K")


def get_settings() -> Settings:
    return Settings()
