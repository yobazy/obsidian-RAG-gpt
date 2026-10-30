import argparse
import os

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.prompt import Prompt

from rag.config import add_common_args, settings_from_args
from rag.pipeline import retrieve, user_message, stream_answer
from rag.providers import ProviderUnavailable, make_provider
from rag.store import make_store

console = Console()


def chat(store, provider, settings):
    conversation_history = []

    console.print("\n[bold green]🧠 Obsidian RAG[/bold green] — Ask anything about your notes")
    console.print(f"[dim]{settings.provider} provider · type 'quit' to exit[/dim]\n")

    while True:
        query = Prompt.ask("[bold cyan]You[/bold cyan]")
        if query.lower() in ("quit", "exit", "q"):
            console.print("\n[dim]Bye![/dim]")
            break

        try:
            chunks = retrieve(store, provider, query, settings.top_k, settings.threshold,
                              **settings.retrieval_options())
            console.print("\n[bold magenta]Assistant[/bold magenta]")
            text = ""
            with Live(Markdown(""), console=console, refresh_per_second=8) as live:
                for token in stream_answer(provider, query, chunks, conversation_history):
                    text += token
                    live.update(Markdown(text))
        except ProviderUnavailable as e:
            console.print(f"[red]{e}[/red]\n")
            continue

        conversation_history.append({"role": "user", "content": user_message(query, chunks)})
        conversation_history.append({"role": "assistant", "content": text})
        console.print()

        console.print("[dim]Sources:[/dim]")
        for chunk in chunks:
            filename = os.path.basename(chunk["file"])
            score = "linked" if chunk["via_link"] else f"{chunk['similarity']:.2f}"
            section = f" › {chunk['heading']}" if chunk.get("heading") else ""
            console.print(f"[dim]  • {filename}{section} ({score})[/dim]")
        console.print()


def main():
    args = add_common_args(argparse.ArgumentParser(description="Chat with your notes")).parse_args()
    settings = settings_from_args(args)
    store = make_store(settings)
    store.setup()
    try:
        chat(store, make_provider(settings), settings)
    finally:
        store.close()


if __name__ == "__main__":
    main()
