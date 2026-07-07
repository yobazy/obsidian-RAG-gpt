import os
from dotenv import load_dotenv
import voyageai
import psycopg2
import anthropic
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

load_dotenv()

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

voyage = voyageai.Client(api_key=VOYAGE_API_KEY)
claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
console = Console()

def retrieve(conn, query, top_k=5):
    """Embed the query and find the most similar chunks."""
    result = voyage.embed([query], model="voyage-3", input_type="query")
    query_embedding = result.embeddings[0]

    with conn.cursor() as cur:
        cur.execute("""
            SELECT file_path, content,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, query_embedding, top_k))
        rows = cur.fetchall()

    return [
        {"file": row[0], "content": row[1], "similarity": row[2]}
        for row in rows
    ]

def build_context(chunks):
    """Format retrieved chunks into a context block for Claude."""
    parts = []
    for i, chunk in enumerate(chunks):
        filename = os.path.basename(chunk["file"])
        parts.append(f"[{i+1}] From '{filename}' (similarity: {chunk['similarity']:.2f}):\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)

def chat(conn):
    conversation_history = []

    console.print("\n[bold green]🧠 Obsidian RAG[/bold green] — Ask anything about your notes\n")
    console.print("[dim]Type 'quit' to exit[/dim]\n")

    while True:
        query = Prompt.ask("[bold cyan]You[/bold cyan]")
        if query.lower() in ("quit", "exit", "q"):
            console.print("\n[dim]Bye![/dim]")
            break

        # Retrieve relevant chunks
        chunks = retrieve(conn, query)
        context = build_context(chunks)

        # Build system prompt
        system_prompt = """You are a personal knowledge assistant with access to the user's Obsidian notes.
Answer questions based on the provided context from their notes.
Be conversational and specific — reference actual content from the notes.
If the context doesn't contain enough information to answer, say so honestly.
Never make up information that isn't in the notes."""

        # Add user message with context
        user_message = f"""Here is relevant context from my notes:

{context}

---

My question: {query}"""

        conversation_history.append({"role": "user", "content": user_message})

        # Call Claude
        response = claude.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system_prompt,
            messages=conversation_history,
        )

        assistant_message = response.content[0].text
        conversation_history.append({"role": "assistant", "content": assistant_message})

        # Print response
        console.print(f"\n[bold magenta]Assistant[/bold magenta]")
        console.print(Markdown(assistant_message))
        console.print()

        # Show sources
        console.print("[dim]Sources:[/dim]")
        for chunk in chunks:
            filename = os.path.basename(chunk["file"])
            console.print(f"[dim]  • {filename} ({chunk['similarity']:.2f})[/dim]")
        console.print()

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        chat(conn)
    finally:
        conn.close()

if __name__ == "__main__":
    main()