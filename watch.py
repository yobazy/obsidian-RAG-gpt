import argparse
import os
import sys
import time

from rich.console import Console
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from rag.config import add_common_args, settings_from_args
from rag.pipeline import in_dot_folder, ingest_file
from rag.providers import make_provider
from rag.store import make_store

console = Console()


class VaultHandler(FileSystemEventHandler):
    def __init__(self, store, provider, vault, chunking="heading"):
        self.store = store
        self.provider = provider
        self.vault = vault
        self.chunking = chunking

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            self._handle(event.src_path)

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            self._handle(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            self._remove(event.src_path)

    def on_moved(self, event):
        # Renames, moves, and editors that save by writing a temp file then renaming it.
        if event.is_directory:
            return
        if event.src_path.endswith(".md"):
            self._remove(event.src_path)
        if event.dest_path.endswith(".md"):
            self._handle(event.dest_path)

    def _remove(self, path):
        try:
            self.store.delete_file(path)
            console.print(f"[yellow]✕ Removed from index:[/yellow] {os.path.basename(path)}")
        except Exception as e:
            console.print(f"[red]❌ Error removing {os.path.basename(path)}: {e}[/red]")

    def _handle(self, path):
        if in_dot_folder(path, self.vault):
            return
        filename = os.path.basename(path)
        console.print(f"[yellow]⟳ Detected change:[/yellow] {filename}")
        try:
            count = ingest_file(self.store, self.provider, path, self.chunking)
            if count:
                console.print(f"[green]✅ Re-ingested:[/green] {filename} → {count} chunks")
            else:
                console.print(f"[dim]↩ No change detected, skipped: {filename}[/dim]")
        except Exception as e:
            console.print(f"[red]❌ Error ingesting {filename}: {e}[/red]")


def main():
    args = add_common_args(argparse.ArgumentParser(description="Re-ingest notes as they change")).parse_args()
    settings = settings_from_args(args)
    if not settings.vault_path:
        sys.exit("Set VAULT_PATH (or pass --vault).")
    store = make_store(settings)
    store.setup()
    vault = str(settings.vault_root)

    handler = VaultHandler(store, make_provider(settings), vault, settings.chunking)
    observer = Observer()
    observer.schedule(handler, vault, recursive=True)
    observer.start()

    console.print(f"\n[bold green]👁 Watching vault:[/bold green] {vault} ({settings.provider} provider)")
    console.print("[dim]Edit any note to trigger re-ingestion. Ctrl+C to stop.[/dim]\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        console.print("\n[dim]Stopped watching.[/dim]")

    observer.join()
    store.close()


if __name__ == "__main__":
    main()
