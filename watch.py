import os
import time
import psycopg2
from dotenv import load_dotenv
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from rich.console import Console
from ingest import ingest_file, setup_db

load_dotenv()

VAULT_PATH = os.getenv("VAULT_PATH")
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

console = Console()

class VaultHandler(FileSystemEventHandler):
    def __init__(self, conn):
        self.conn = conn

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            self._handle(event.src_path)

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            self._handle(event.src_path)

    def _handle(self, path):
        filename = os.path.basename(path)
        console.print(f"[yellow]⟳ Detected change:[/yellow] {filename}")
        try:
            count = ingest_file(self.conn, path)
            if count:
                console.print(f"[green]✅ Re-ingested:[/green] {filename} → {count} chunks")
            else:
                console.print(f"[dim]↩ No change detected, skipped: {filename}[/dim]")
        except Exception as e:
            console.print(f"[red]❌ Error ingesting {filename}: {e}[/red]")

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    setup_db(conn)

    handler = VaultHandler(conn)
    observer = Observer()
    observer.schedule(handler, VAULT_PATH, recursive=True)
    observer.start()

    console.print(f"\n[bold green]👁 Watching vault:[/bold green] {VAULT_PATH}")
    console.print("[dim]Edit any note to trigger re-ingestion. Ctrl+C to stop.[/dim]\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        console.print("\n[dim]Stopped watching.[/dim]")

    observer.join()
    conn.close()

if __name__ == "__main__":
    main()
