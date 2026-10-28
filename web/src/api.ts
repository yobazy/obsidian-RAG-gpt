export type NoteRef = { name: string; path: string | null };

export type Source = {
  n: number;
  path: string;
  title: string;
  heading: string | null;
  score: number | null;
  via_link: boolean;
  snippet: string;
  obsidian_url: string;
  links: NoteRef[];
  backlinks: NoteRef[];
};

export type Status = {
  provider: "cloud" | "local";
  embed_model: string;
  chat_model: string;
  store: string;
  vault_name: string;
  index: { notes: number; chunks: number };
  last_ingest: { finished_at: string; files_seen: number; chunks_written: number } | null;
  indexing: boolean;
  index_error: string | null;
  backend: { reachable: boolean; error?: string; embed_loaded?: boolean; chat_loaded?: boolean };
};

export type Note = {
  path: string;
  title: string;
  frontmatter: Record<string, unknown>;
  body: string;
  obsidian_url: string;
  links: NoteRef[];
  backlinks: NoteRef[];
  indexed: boolean;
};

export type AskEvent =
  | { type: "status"; message: string }
  | { type: "sources"; sources: Source[]; retrieval_ms: number }
  | { type: "token"; text: string }
  | { type: "done"; first_token_ms: number | null; total_ms: number }
  | { type: "error"; code: string; message: string };

export type Turn = { role: "user" | "assistant"; content: string };

async function json<T>(res: Response): Promise<T> {
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error ?? `Request failed (${res.status})`);
  return body as T;
}

export const getStatus = () => fetch("/status").then((r) => json<Status>(r));

export const getNote = (path: string) =>
  fetch("/notes/" + path.split("/").map(encodeURIComponent).join("/")).then((r) => json<Note>(r));

/** POST /ask and hand each NDJSON event to onEvent as it arrives. */
export async function ask(
  question: string,
  history: Turn[],
  onEvent: (event: AskEvent) => void,
  signal: AbortSignal,
) {
  const res = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history }),
    signal,
  });
  if (!res.ok || !res.body) {
    await json(res);
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newline;
    while ((newline = buffer.indexOf("\n")) !== -1) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (line) onEvent(JSON.parse(line) as AskEvent);
    }
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer) as AskEvent);
}
