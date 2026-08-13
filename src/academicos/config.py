"""Runtime configuration: paths, corpus layout, sync target, provider flags.

Config is read from config/config.toml (project-local) and env vars. TOML
values can be overridden by uppercase env vars prefixed ACOS_ (e.g. ACOS_DATA_ROOT).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import tomllib

_HERE = Path(__file__).resolve().parent.parent.parent
SECRETS_FILE = _HERE / "config" / "secrets.env"


def load_secrets(path: Path | None = None) -> int:
    """Load KEY=VALUE lines from the gitignored secrets file into the environment.

    Kept out of config.toml on purpose: that file is committed, and credentials
    in it end up in git history. Real environment variables always win, so a
    deployment can inject secrets without this file existing at all.
    """
    p = path or SECRETS_FILE
    if not p.exists():
        return 0
    loaded = 0
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


class Config:
    def __init__(self, data_root: Path | None = None, toml: dict[str, Any] | None = None):
        t = toml or {}
        env = lambda k, d=None: os.environ.get(f"ACOS_{k.upper()}", d)

        self.data_root = Path(env("DATA_ROOT", str(data_root or Path.cwd() / "academicos-data")))
        self.corpus_root = Path(env("CORPUS_ROOT", t.get("corpus_root", str(self.data_root / "corpus"))))
        self.gcs_bucket = env("GCS_BUCKET", t.get("gcs_bucket", ""))
        self.gcs_prefix = env("GCS_PREFIX", t.get("gcs_prefix", "academicos"))
        sync = env("SYNC_ENABLED", t.get("sync_enabled", "false"))
        self.sync_enabled = sync if isinstance(sync, bool) else str(sync).lower() == "true"

        self.documents_dir = self.data_root / "documents"
        self.pages_dir = self.data_root / "pages"
        self.parse_dir = self.data_root / "parse"
        self.extracted_dir = self.data_root / "extracted"
        self.graph_db = self.data_root / "graph" / "graph.sqlite"
        self.index_db = self.data_root / "index" / "index.sqlite"
        self.registry_db = self.data_root / "registry" / "registry.sqlite"
        self.events_db = self.data_root / "events" / "events.sqlite"
        self.question_map_db = self.data_root / "questions" / "question-maps.jsonl"
        self.audit_log = self.data_root / "governance" / "audit.jsonl"
        self.artifacts_dir = self.data_root / "artifacts"

        for d in (
            self.documents_dir, self.pages_dir, self.parse_dir, self.extracted_dir,
            self.data_root / "graph", self.data_root / "index", self.data_root / "registry",
            self.data_root / "events", self.data_root / "governance", self.artifacts_dir,
            self.question_map_db.parent,
        ):
            d.mkdir(parents=True, exist_ok=True)

        # Parsing strategy
        self.parse_priority = t.get("parse_priority", ["pdf_native", "unlimited_ocr", "tesseract"])
        self.vlm_device = env("VLM_DEVICE", t.get("vlm_device", "cuda" if os.environ.get("CUDA_AVAILABLE") else "cuda"))
        self.min_confidence = float(t.get("min_confidence", 0.6))
        self.max_pages_per_doc = int(t.get("max_pages_per_doc", 500))

        # LLM provider (OpenIE extraction + Self-RAG-style critic)
        self.llm_base_url = env("LLM_BASE_URL", t.get("llm_base_url", "https://api.sarvam.ai"))
        self.llm_api_key = env("LLM_API_KEY", t.get("llm_api_key", ""))
        self.llm_model = env("LLM_MODEL", t.get("llm_model", "sarvam-105b"))
        self.llm_timeout = float(t.get("llm_timeout", 120))
        self.openie_max_chunk_chars = int(t.get("openie_max_chunk_chars", 1800))
        self.critic_retrieve_threshold = float(t.get("critic_retrieve_threshold", 0.5))
        self.critic_weights = {
            "w_rel": float(t.get("critic_w_rel", 1.0)),
            "w_sup": float(t.get("critic_w_sup", 1.0)),
            "w_use": float(t.get("critic_w_use", 0.5)),
        }

    @classmethod
    def load(cls, toml_path: Path | None = None) -> "Config":
        load_secrets()
        toml: dict[str, Any] = {}
        p = toml_path or (_HERE / "config" / "config.toml")
        if p.exists():
            toml = tomllib.loads(p.read_text(encoding="utf-8"))
        return cls(toml=toml)


def get_config() -> Config:
    return Config.load()
