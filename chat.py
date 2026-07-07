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

SYSTEM_PROMPT = """You are a personal knowledge assistant with access to the user's Obsidian notes.
Answer questions based on the provided context from their notes.
Be conversational and specific — reference actual content from the notes.
If the context doesn't contain enough information to answer, say so honestly.
Never make up information that isn't in the notes."""

def retrieve(conn, query, top_k=5, min_similarity=0.35):
    """Embed the query, find similar chunks, expand via backlinks."""
    result = voyage.embed([query], model="voyage-3", input_type="query")
    query_embedding = result.embeddings[0]

    with conn.cursor() as cur:
        cur.execute("""
            SELECT file_path, content, backlinks,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM chunks
            WHERE 1 - (embedding <=> %s::vector) > %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, query_embedding, min_similarity, query_embedding, top_k))
        primary = cur.fetchall()

        if not primary:
            return []

        linked_names = set()
        for row in primary:
            for link in (row[2] or []):
                linked_names.add(link.lower())

        linked_chunks = []
        if linked_names:
            cur.execute("""
                SELECT file_path, content, backlinks, 0.0 AS similarity
                FROM chunks
                WHERE LOWER(REPLACE(SPLIT_PART(file_path, '/', -1), '.md', '')) = ANY(%s)
                LIMIT 10
            """, (list(linked_names),))
            linked_chunks = cur.fetchall()

    seen = set()
    results = []
    for row in list(primary) + linked_chunks:
        key = (row[0], row[1][:50])
        if key not in seen:
            seen.add(key)
            results.append({
                "file": row[0],
                "content": row[1],
                "similarity": row[3],
            })

    return results

def build_context(chunks):
    """Format retrieved chunks into a context block for Claude."""
    parts = []
    for i, chunk in enumerate(chunks):
        filename = os.path.basename(chunk["file"])
        parts.append(f"[{i+1}] From '{filename}' (similarity: {chunk['similarity']:.2f}):\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)

def answer(query, chunks, history=None):
    """Generate an answer from retrieved chunks."""
    context = build_context(chunks)
    user_message = f"""Here is relevant context from my notes:

{context}

---

My question: {query}"""

    messages = list(history or [])
    messages.append({"role": "user", "content": user_message})

    response = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    text = response.content[0].text

    if history is not None:
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": text})

    return text

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
        assistant_message = answer(query, chunks, conversation_history)

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