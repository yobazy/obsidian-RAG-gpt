"""Settings, read from the environment (and .env when python-dotenv is installed).

RAG_PROVIDER picks the backend:
  cloud  Voyage AI embeddings + Claude answers (the original setup)
  local  Ollama on the home server: nomic-embed-text + qwen3:8b

Each provider keeps its own chunks table because the embedding sizes differ
(Voyage 1024, nomic 768). The cloud table keeps its original name, `chunks`,
so existing indexes carry over untouched.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional; plain env vars work too
    pass

PROVIDERS = ("cloud", "local")


@dataclass
class ProviderSpec:
    name: str
    embed_model: str
    chat_model: str
    dim: int
    table: str
    min_similarity: float


@dataclass
class Settings:
    provider: str = "cloud"
    vault_path: str | None = None
    vault_name: str = "baz-vault"
    store: str = "postgres"  # postgres | memory
    db: dict = field(default_factory=dict)

    ollama_url: str = "https://pc.taila3582d.ts.net:11434"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_chat_model: str = "qwen3:8b"
    ollama_num_ctx: int = 8192
    # First call after an idle spell loads the model, which takes seconds.
    ollama_timeout: float = 180.0
    ollama_connect_timeout: float = 10.0

    voyage_model: str = "voyage-3"
    claude_model: str = "claude-sonnet-4-6"

    top_k: int = 5
    min_similarity: float | None = None  # None means the provider default

    def spec(self) -> ProviderSpec:
        if self.provider == "local":
            return ProviderSpec(
                name="local",
                embed_model=self.ollama_embed_model,
                chat_model=self.ollama_chat_model,
                dim=768,
                table=os.getenv("CHUNKS_TABLE_LOCAL", "chunks_nomic_768"),
                min_similarity=0.5,
            )
        return ProviderSpec(
            name="cloud",
            embed_model=self.voyage_model,
            chat_model=self.claude_model,
            dim=1024,
            table=os.getenv("CHUNKS_TABLE_CLOUD", "chunks"),
            min_similarity=0.35,
        )

    @property
    def threshold(self) -> float:
        if self.min_similarity is not None:
            return self.min_similarity
        return self.spec().min_similarity

    @property
    def vault_root(self) -> Path | None:
        # Not resolved: stored paths must match what earlier ingests wrote.
        return Path(self.vault_path).expanduser() if self.vault_path else None


def _float(name, default):
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default


def add_common_args(parser):
    """--provider / --vault flags shared by the CLI scripts."""
    parser.add_argument("--provider", choices=PROVIDERS, help="Overrides RAG_PROVIDER")
    parser.add_argument("--vault", help="Overrides VAULT_PATH")
    return parser


def settings_from_args(args) -> Settings:
    overrides = {}
    if getattr(args, "provider", None):
        overrides["provider"] = args.provider
    if getattr(args, "vault", None):
        overrides["vault_path"] = args.vault
    return load_settings(**overrides)


def load_settings(**overrides) -> Settings:
    provider = overrides.get("provider") or os.getenv("RAG_PROVIDER", "cloud").strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"RAG_PROVIDER must be one of {PROVIDERS}, got {provider!r}")
    settings = Settings(
        provider=provider,
        vault_path=os.getenv("VAULT_PATH"),
        vault_name=os.getenv("VAULT_NAME", "baz-vault"),
        store=os.getenv("RAG_STORE", "postgres"),
        db={
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT"),
            "dbname": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
        },
        ollama_url=os.getenv("OLLAMA_URL", Settings.ollama_url).rstrip("/"),
        ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", Settings.ollama_embed_model),
        ollama_chat_model=os.getenv("OLLAMA_CHAT_MODEL", Settings.ollama_chat_model),
        ollama_num_ctx=int(os.getenv("OLLAMA_NUM_CTX", Settings.ollama_num_ctx)),
        ollama_timeout=_float("OLLAMA_TIMEOUT", Settings.ollama_timeout),
        voyage_model=os.getenv("VOYAGE_MODEL", Settings.voyage_model),
        claude_model=os.getenv("CLAUDE_MODEL", Settings.claude_model),
        top_k=int(os.getenv("RAG_TOP_K", Settings.top_k)),
        min_similarity=_float("RAG_MIN_SIMILARITY", None),
    )
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings
