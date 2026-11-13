from pathlib import Path

from mind_recsys.config import Settings, get_settings


def test_default_settings():
    settings = Settings(_env_file=None)
    assert settings.qdrant_host == "localhost"
    assert settings.qdrant_port == 6333
    assert settings.embedding_model_name == "BAAI/bge-small-en-v1.5"
    assert settings.candidate_top_k == 100
    assert settings.ranker_top_k == 10
    assert settings.mind_data_dir == Path("./data")
    assert settings.artifacts_dir == Path("./artifacts")


def test_custom_env_override(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_secret_token_12345")
    monkeypatch.setenv("QDRANT_PORT", "6335")
    monkeypatch.setenv("CANDIDATE_TOP_K", "50")

    settings = Settings(_env_file=None)
    assert settings.hf_token == "hf_test_secret_token_12345"
    assert settings.qdrant_port == 6335
    assert settings.candidate_top_k == 50


def test_get_settings_singleton_call():
    settings = get_settings()
    assert isinstance(settings, Settings)
