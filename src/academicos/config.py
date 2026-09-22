"""Runtime configuration: paths, corpus layout, sync target, provider flags.

Config is read from config/config.toml (project-local) and env vars. TOML
values can be overridden by uppercase env vars prefixed ACOS_ (e.g. ACOS_DATA_ROOT).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import tomllib

_HERE = Path(__file__).resolve().parent.parent.parent
SECRETS_FILE = _HERE / "config" / "secrets.env"

# The value `deploy/gcp/bootstrap.sh` writes into every Secret Manager entry it
# creates, so Cloud Run can start before the team has pasted real credentials.
# It must therefore read as "not configured": it is a non-empty string, so a
# plain `bool(url and key)` would call it configured, and the blob stores
# (scan media at assessment/mobile_scan.py, curriculum snapshots at
# curriculum/store.py) would then try to reach the host `REPLACE_ME` and fail
# with a DNS error on a deployment that otherwise looks perfectly healthy.
SUPABASE_PLACEHOLDER = "REPLACE_ME"


def credential_or_none(name: str) -> Optional[str]:
    """Read a credential from the environment, treating an empty value and the
    deployment placeholder alike as 'not configured'.

    The single source of truth for that rule -- `assessment/supabase_kv.py`
    imports this rather than repeating the comparison.
    """
    raw = os.environ.get(name, "").strip()
    if not raw or raw == SUPABASE_PLACEHOLDER:
        return None
    return raw


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
        self.artifacts_dir = self.data_root / "artifacts"
        self.chapter_tags_db = self.data_root / "assessment" / "chapter_tags.sqlite"

        for d in (
            self.documents_dir, self.pages_dir, self.parse_dir, self.extracted_dir,
            self.data_root / "graph", self.data_root / "index", self.data_root / "registry",
            self.data_root / "events", self.artifacts_dir,
            self.question_map_db.parent, self.chapter_tags_db.parent,
        ):
            d.mkdir(parents=True, exist_ok=True)

        # Parsing strategy
        self.parse_priority = t.get("parse_priority", ["pdf_native", "unlimited_ocr", "tesseract"])
        self.vlm_device = env("VLM_DEVICE", t.get("vlm_device", "cuda" if os.environ.get("CUDA_AVAILABLE") else "cuda"))
        self.min_confidence = float(t.get("min_confidence", 0.6))
        self.max_pages_per_doc = int(t.get("max_pages_per_doc", 500))

        # LLM provider (OpenIE extraction + Self-RAG-style critic)
        self.llm_base_url = env("LLM_BASE_URL", t.get("llm_base_url", "https://api.sarvam.ai"))
        # bootstrap.sh fills the acos-llm-api-key secret with REPLACE_ME, so the
        # placeholder must read as "no key" -- otherwise the critic is built
        # and every doubt sends a request that fails with a 401.
        self.llm_api_key = credential_or_none("ACOS_LLM_API_KEY") or t.get("llm_api_key", "")
        self.llm_model = env("LLM_MODEL", t.get("llm_model", "sarvam-105b"))
        self.llm_timeout = float(t.get("llm_timeout", 120))
        self.openie_max_chunk_chars = int(t.get("openie_max_chunk_chars", 1800))
        self.critic_retrieve_threshold = float(t.get("critic_retrieve_threshold", 0.5))
        self.critic_weights = {
            "w_rel": float(t.get("critic_w_rel", 1.0)),
            "w_sup": float(t.get("critic_w_sup", 1.0)),
            "w_use": float(t.get("critic_w_use", 0.5)),
        }

        # Principal-role bootstrap key (see assessment/users.py). Secret --
        # belongs in the gitignored config/secrets.env via
        # ACOS_PRINCIPAL_BOOTSTRAP_KEY, never in config.toml. Empty by
        # default, which is the secure default: no key configured means no
        # self-registration can ever be granted the principal role.
        self.principal_bootstrap_key = env("PRINCIPAL_BOOTSTRAP_KEY", t.get("principal_bootstrap_key", ""))

        # CORS origins: comma-separated list of allowed origins or '*'.
        # Defaults to local dev ports and trusted deployment origins.
        cors_raw = env("CORS_ORIGINS", t.get("cors_origins", "")).strip()
        if cors_raw:
            self.cors_origins = [o.strip() for o in cors_raw.split(",") if o.strip()]
        else:
            self.cors_origins = [
                "http://localhost:3000",
                "http://localhost:8000",
                "http://localhost:8080",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:8000",
                "http://127.0.0.1:8080",
                "https://sarvamai.github.io",
            ]

        # Supabase (hosted Postgres), used by the operational stores under
        # assessment/ (and now storage/event_store.py, curriculum/store.py)
        # as a durability backstop -- Render's disk is ephemeral, so anything
        # meant to survive a redeploy has to live here instead. The actual
        # HTTP client (SupabaseTable/SupabaseStorage in
        # assessment/supabase_kv.py) reads these same two env vars directly
        # -- these Config fields exist purely so CLI `stats` and the /health
        # endpoint can report whether Supabase is configured. The placeholder
        # rule lives in `credential_or_none` so config.py and supabase_kv.py
        # cannot drift apart on what "configured" means.
        self.supabase_url = credential_or_none("SUPABASE_KNOWLEDGE_URL")
        self.supabase_enabled = bool(
            self.supabase_url and credential_or_none("SUPABASE_KNOWLEDGE_ANON_KEY")
        )

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


# --------------------------------------------------------------------------- #
# Production configuration guard (the GCP release)
# --------------------------------------------------------------------------- #
# The 2026-09-21 audit (item 8.3) found a release nobody could identify and a
# bootstrap that creates every secret as REPLACE_ME. A service started on the
# placeholder, with CORS open to "*" or with the demo principal seeded, came up
# answering /health 200. In production these are refused at startup instead:
# a revision that will not start fails the deploy, a revision that starts
# wrong serves a school.

log = logging.getLogger(__name__)

# The shortest principal bootstrap key production accepts. The key is the
# only thing that turns a self-registration into a principal account, so it
# gets the length of a random 128-bit token in hex, not of a password.
MIN_PRINCIPAL_BOOTSTRAP_KEY_CHARS = 32

# Secrets production cannot run without. The LLM key is deliberately NOT here:
# a school without AI features is a valid deployment (features answer "not
# enabled for this school", see LLMNotEnabled).
REQUIRED_PRODUCTION_SECRETS = ("ACOS_PRINCIPAL_BOOTSTRAP_KEY", "ACOS_POSTGRES_PASSWORD")

# Hosts an LLM feature may send school data to. Must equal the "LLM providers"
# table in docs/compliance.md (tests/test_production_guard.py pins the two
# together -- docs/ is not in the image, so the list lives here too). Adding a
# provider means adding it there, here, and to the DPA's sub-processor list.
APPROVED_LLM_HOSTS = frozenset({"api.sarvam.ai"})

LLM_NOT_ENABLED_DETAIL = (
    "AI features (scan reading, AI marking) are not enabled for this school. "
    "Ask your administrator to configure an AI provider key."
)

_TRUTHY = {"1", "true", "yes", "on"}


class ProductionConfigError(RuntimeError):
    """Startup refused: the production configuration is unsafe or incomplete."""


class LLMNotEnabled(RuntimeError):
    """An LLM feature was asked for on a deployment with no provider key.

    Not a subclass of VisionError / LLMEvaluationError on purpose: the scan
    pipeline catches those per page / per question and records a blank page or
    a needs-review mark, which turned "no key" into a booklet of zeros. This
    one propagates to the API as a clear 501.
    """

    def __init__(self, message: str = LLM_NOT_ENABLED_DETAIL):
        super().__init__(message)


def is_production() -> bool:
    """True on the GCP release.

    Explicit `ACOS_ENV=production` (deploy-gcp.yml sets it), or Cloud Run's
    Cloud SQL unix-socket mount (`ACOS_POSTGRES_HOST=/cloudsql/...`), which is
    how durable_table reaches Cloud SQL there and which no local, dev or CI
    environment uses -- so the guard still holds if ACOS_ENV is ever dropped
    from the deploy. A local Postgres (`localhost`) is not production.
    """
    if os.environ.get("ACOS_ENV", "").strip().lower() == "production":
        return True
    return os.environ.get("ACOS_POSTGRES_HOST", "").strip().startswith("/cloudsql/")


def demo_accounts_enabled() -> bool:
    """Whether UserStore may seed the demo teacher/principal (password123).

    `ACOS_DEMO_ACCOUNTS` decides when set; otherwise on everywhere except
    production, so local development keeps its ready-made logins.
    """
    raw = os.environ.get("ACOS_DEMO_ACCOUNTS", "").strip().lower()
    if raw:
        return raw in _TRUTHY
    return not is_production()


def llm_key_configured() -> bool:
    return bool(credential_or_none("SARVAM_API_KEY") or credential_or_none("ACOS_LLM_API_KEY"))


def _llm_feature_urls(cfg: "Config") -> list[tuple[str, str]]:
    """Every base URL an LLM feature sends school data to. Imported lazily:
    these modules import config, not the other way round."""
    from .assessment import llm_evaluate, vision
    from .llm import sarvam

    return [
        ("doubt critic (ACOS_LLM_BASE_URL / config.toml llm_base_url)", cfg.llm_base_url),
        ("scan OCR (assessment/vision.py)", vision.BASE_URL),
        ("AI marking (assessment/llm_evaluate.py)", llm_evaluate.CHAT_URL),
        ("curriculum extraction (llm/sarvam.py)", sarvam.DEFAULT_BASE_URL),
    ]


def production_config_problems(cfg: "Config") -> list[str]:
    """Everything that makes this configuration unfit to serve a school.
    Empty outside production: local, dev and test are never refused."""
    if not is_production():
        return []
    problems: list[str] = []
    for name in REQUIRED_PRODUCTION_SECRETS:
        if credential_or_none(name) is None:
            problems.append(f"{name} is empty or still the bootstrap placeholder "
                            f"{SUPABASE_PLACEHOLDER!r}; add a real Secret Manager version")
    key = cfg.principal_bootstrap_key or ""
    if key and key != SUPABASE_PLACEHOLDER and len(key) < MIN_PRINCIPAL_BOOTSTRAP_KEY_CHARS:
        problems.append(f"ACOS_PRINCIPAL_BOOTSTRAP_KEY is {len(key)} characters; production "
                        f"needs at least {MIN_PRINCIPAL_BOOTSTRAP_KEY_CHARS} "
                        "(e.g. `openssl rand -hex 32`)")
    if "*" in cfg.cors_origins:
        problems.append("ACOS_CORS_ORIGINS allows '*'; list the web origins "
                        "(https://<project>.web.app,https://<project>.firebaseapp.com)")
    else:
        insecure = [o for o in cfg.cors_origins if not o.startswith("https://")]
        if insecure:
            problems.append(f"CORS origins include non-https {insecure} -- ACOS_CORS_ORIGINS is "
                            "unset (dev defaults) or wrong; set it to the Hosting origins")
    if demo_accounts_enabled():
        problems.append("ACOS_DEMO_ACCOUNTS is on: the demo principal (password123) would be "
                        "seeded; unset it")
    if os.environ.get("ACOS_SEED_DEMO_USERS", "").strip().lower() in ("1", "true", "yes"):
        problems.append("ACOS_SEED_DEMO_USERS is set: it asks for the password123 demo accounts, "
                        "which a school's service must never have; unset it")
    if llm_key_configured():
        from urllib.parse import urlparse

        for feature, url in _llm_feature_urls(cfg):
            host = (urlparse(url).hostname or "").lower()
            if host not in APPROVED_LLM_HOSTS:
                problems.append(f"{feature} sends to {host or url!r}, which is not in "
                                "docs/compliance.md's LLM provider table "
                                f"({', '.join(sorted(APPROVED_LLM_HOSTS))}); set "
                                "ACOS_LLM_BASE_URL to a listed provider or list this one")
    return problems


def enforce_production_config(cfg: "Config") -> None:
    """Raise (after logging each problem on its own line) if production is
    misconfigured. Called first thing at startup, before any store opens."""
    problems = production_config_problems(cfg)
    if not problems:
        return
    for p in problems:
        log.error("REFUSING TO START (production config): %s", p)
    raise ProductionConfigError("production configuration refused: " + "; ".join(problems))


def build_identity() -> dict[str, str]:
    """What build this process is. Baked into the image by the Dockerfile's
    BUILD_COMMIT / BUILT_AT build args; 'unknown' for a local run. Read per
    call, not at import, so it is exactly what the environment says."""
    return {
        "commit": os.environ.get("ACOS_BUILD_COMMIT", "").strip() or "unknown",
        "builtAt": os.environ.get("ACOS_BUILT_AT", "").strip() or "unknown",
    }
