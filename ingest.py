import argparse
import sys

from rag.config import add_common_args, settings_from_args
from rag.pipeline import ingest_vault
from rag.providers import ProviderUnavailable, make_provider
from rag.store import make_store


def main():
    parser = add_common_args(argparse.ArgumentParser(description="Index the vault into pgvector"))
    args = parser.parse_args()
    settings = settings_from_args(args)
    if not settings.vault_path:
        sys.exit("Set VAULT_PATH (or pass --vault).")
    if settings.store == "memory":
        sys.exit("RAG_STORE=memory keeps the index in the server process; ingest.py needs Postgres.")

    store = make_store(settings)
    store.setup()
    print(f"✅ DB ready ({settings.provider} provider, table {store.table})")
    try:
        ingest_vault(store, make_provider(settings), settings.vault_root)
    except ProviderUnavailable as e:
        sys.exit(f"❌ {e}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
