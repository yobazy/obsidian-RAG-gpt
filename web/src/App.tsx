import { useCallback, useEffect, useRef, useState } from "react";
import { ask, getNote, getStatus, type Note, type NoteRef, type Source, type Status, type Turn } from "./api";
import { Markdown } from "./Markdown";

type Exchange = {
  id: number;
  question: string;
  answer: string;
  sources: Source[];
  state: "searching" | "answering" | "done" | "stopped" | "error";
  notice?: string;
  error?: string;
  timings?: { retrieval_ms?: number; first_token_ms?: number | null; total_ms?: number };
};

const HISTORY_TURNS = 3;

export default function App() {
  const [status, setStatus] = useState<Status | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  const [preview, setPreview] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);
  const nextId = useRef(1);

  const refreshStatus = useCallback(() => {
    getStatus()
      .then((s) => {
        setStatus(s);
        setStatusError(null);
      })
      .catch(() => setStatusError("The API server isn't answering. Start it with python3 server.py."));
  }, []);

  useEffect(() => {
    refreshStatus();
    const timer = setInterval(refreshStatus, 30_000);
    return () => clearInterval(timer);
  }, [refreshStatus]);

  const closePreview = useCallback(() => setPreview(null), []);
  const busy = exchanges.some((e) => e.state === "searching" || e.state === "answering");

  const submit = async (question: string) => {
    abort.current?.abort();
    const controller = new AbortController();
    abort.current = controller;
    const id = nextId.current++;
    const history: Turn[] = exchanges
      .filter((e) => e.state === "done")
      .slice(-HISTORY_TURNS)
      .flatMap((e) => [
        { role: "user" as const, content: e.question },
        { role: "assistant" as const, content: e.answer },
      ]);
    const update = (patch: (e: Exchange) => Partial<Exchange>) =>
      setExchanges((all) => all.map((e) => (e.id === id ? { ...e, ...patch(e) } : e)));

    setExchanges((all) => [...all, { id, question, answer: "", sources: [], state: "searching" }]);
    try {
      await ask(
        question,
        history,
        (event) => {
          if (event.type === "status") update(() => ({ notice: event.message }));
          if (event.type === "sources")
            update((e) => ({ sources: event.sources, state: "answering", timings: { ...e.timings, retrieval_ms: event.retrieval_ms } }));
          if (event.type === "token") update((e) => ({ answer: e.answer + event.text, notice: undefined }));
          if (event.type === "done")
            update((e) => ({ state: "done", notice: undefined, timings: { ...e.timings, ...event } }));
          if (event.type === "error") update(() => ({ state: "error", error: event.message, notice: undefined }));
        },
        controller.signal,
      );
      // The stream closed without a done or error event, so the answer is incomplete.
      update((e) =>
        e.state === "done" || e.state === "error"
          ? {}
          : { state: "error", error: "The answer stopped early. Try asking again.", notice: undefined },
      );
    } catch (err) {
      // Stopped answers stay on screen but are left out of the history sent with follow-ups.
      if (controller.signal.aborted) update(() => ({ state: "stopped", notice: undefined }));
      else update(() => ({ state: "error", error: (err as Error).message, notice: undefined }));
    } finally {
      refreshStatus();
    }
  };

  return (
    <div className="page">
      <header className="masthead">
        <div className="wordmark">
          <h1>Ask your notes</h1>
          <p className="dek">Answers drawn from what you wrote, with the notes they came from.</p>
        </div>
        <StatusLine status={status} error={statusError} />
      </header>

      <main>
        {exchanges.length === 0 ? <Empty status={status} /> : null}
        {exchanges.map((exchange) => (
          <ExchangeView key={exchange.id} exchange={exchange} onOpen={setPreview} />
        ))}
        <AskBox busy={busy} onSubmit={submit} onStop={() => abort.current?.abort()} autoFocus={exchanges.length === 0} />
      </main>

      {preview ? <NotePreview path={preview} onOpen={setPreview} onClose={closePreview} /> : null}
    </div>
  );
}

function StatusLine({ status, error }: { status: Status | null; error: string | null }) {
  if (error) {
    return (
      <div className="status" role="status">
        <span className="dot dot-down" aria-hidden /> {error}
      </div>
    );
  }
  if (!status) return <div className="status" role="status">Checking the index…</div>;

  const backend = status.backend;
  const local = status.provider === "local";
  let health: { cls: string; text: string };
  if (!backend.reachable) health = { cls: "dot-down", text: local ? "pc unreachable" : backend.error ?? "Unavailable" };
  else if (local && !backend.chat_loaded) health = { cls: "dot-cold", text: "Model asleep, first answer is slower" };
  else health = { cls: "dot-up", text: "Ready" };

  return (
    <dl className="status" role="status">
      <div>
        <dt>Provider</dt>
        <dd>{local ? "Local" : "Cloud"} · {status.chat_model}</dd>
      </div>
      <div>
        <dt>Index</dt>
        <dd>
          {status.indexing
            ? "Indexing…"
            : `${status.index.notes.toLocaleString()} notes, ${status.index.chunks.toLocaleString()} chunks`}
        </dd>
      </div>
      <div>
        <dt>Last ingest</dt>
        <dd>{status.last_ingest ? ago(status.last_ingest.finished_at) : "Never"}</dd>
      </div>
      <div>
        <dt>Backend</dt>
        <dd>
          <span className={`dot ${health.cls}`} aria-hidden /> {health.text}
        </dd>
      </div>
      {status.index_error ? <p className="status-error">{status.index_error}</p> : null}
    </dl>
  );
}

function Empty({ status }: { status: Status | null }) {
  return (
    <section className="empty">
      <p className="empty-lede">Ask a question the way you'd ask yourself.</p>
      <p className="muted">
        {status && status.index.notes > 0
          ? `Searching ${status.index.notes.toLocaleString()} notes in ${status.vault_name}. Every answer lists the passages it used, and each one opens in Obsidian.`
          : "Nothing is indexed yet. Run python3 ingest.py, then come back."}
      </p>
    </section>
  );
}

function AskBox({
  busy,
  onSubmit,
  onStop,
  autoFocus,
}: {
  busy: boolean;
  onSubmit: (q: string) => void;
  onStop: () => void;
  autoFocus: boolean;
}) {
  const [value, setValue] = useState("");
  const send = () => {
    const q = value.trim();
    if (!q || busy) return;
    onSubmit(q);
    setValue("");
  };
  return (
    <form
      className="askbox"
      onSubmit={(e) => {
        e.preventDefault();
        send();
      }}
    >
      <label htmlFor="question" className="sr-only">Your question</label>
      <textarea
        id="question"
        rows={2}
        value={value}
        autoFocus={autoFocus}
        placeholder="What did I decide about…"
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            send();
          }
        }}
      />
      {busy ? (
        <button type="button" className="button button-quiet" onClick={onStop}>Stop</button>
      ) : (
        <button type="submit" className="button" disabled={!value.trim()}>Ask</button>
      )}
    </form>
  );
}

function ExchangeView({ exchange, onOpen }: { exchange: Exchange; onOpen: (path: string) => void }) {
  const [active, setActive] = useState<number | null>(null);
  const cite = (n: number) => {
    setActive(n);
    document.getElementById(`src-${exchange.id}-${n}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  };
  const citable = new Set(exchange.sources.map((s) => s.n));
  const t = exchange.timings;

  return (
    <article className="exchange">
      <div className="answer-col">
        <h2 className="question">{exchange.question}</h2>
        {exchange.notice ? <p className="notice">{exchange.notice}</p> : null}
        {exchange.state === "searching" && !exchange.notice ? <p className="muted pulse">Searching your notes…</p> : null}
        {exchange.state === "answering" && !exchange.answer && !exchange.notice ? (
          <p className="muted pulse">Writing an answer…</p>
        ) : null}
        <div className="answer" aria-live="polite">
          <Markdown text={exchange.answer} onCite={cite} citable={citable} />
          {exchange.state === "answering" && exchange.answer ? <span className="caret" aria-hidden /> : null}
        </div>
        {exchange.state === "error" ? (
          <div className="error" role="alert">
            <p>{exchange.error}</p>
          </div>
        ) : null}
        {exchange.state === "stopped" ? <p className="timing">Stopped</p> : null}
        {exchange.state === "done" && t?.total_ms ? (
          <p className="timing">
            Found sources in {fmtMs(t.retrieval_ms)}, first words after {fmtMs(t.first_token_ms ?? undefined)}, done in {fmtMs(t.total_ms)}
          </p>
        ) : null}
      </div>

      <aside className="sources-col" aria-label="Sources">
        {exchange.sources.length ? (
          <>
            <h3 className="col-label">Sources</h3>
            <ol className="sources">
              {exchange.sources.map((s) => (
                <SourceCard
                  key={s.n}
                  id={`src-${exchange.id}-${s.n}`}
                  source={s}
                  active={active === s.n}
                  onOpen={onOpen}
                />
              ))}
            </ol>
          </>
        ) : exchange.state !== "searching" ? (
          <p className="muted small">No note was close enough to this question.</p>
        ) : null}
      </aside>
    </article>
  );
}

function SourceCard({
  id,
  source,
  active,
  onOpen,
}: {
  id: string;
  source: Source;
  active: boolean;
  onOpen: (path: string) => void;
}) {
  const showHeading = source.heading && source.heading !== source.title;
  return (
    <li id={id} className={`source${active ? " source-active" : ""}`}>
      <div className="source-top">
        <span className="source-n">{source.n}</span>
        <span className="source-score">
          {source.via_link ? "Linked note" : `${Math.round((source.score ?? 0) * 100)}% match`}
        </span>
      </div>
      <button className="source-title" onClick={() => onOpen(source.path)}>
        {source.title}
        {showHeading ? <span className="source-heading"> › {source.heading}</span> : null}
      </button>
      <p className="source-path">{source.path}</p>
      <p className="source-snippet">{source.snippet}</p>
      <LinkRow label="Links to" refs={source.links} onOpen={onOpen} />
      <LinkRow label="Linked from" refs={source.backlinks} onOpen={onOpen} />
      <div className="source-actions">
        <button className="text-button" onClick={() => onOpen(source.path)}>Preview</button>
        <a className="text-button" href={source.obsidian_url}>Open in Obsidian</a>
      </div>
    </li>
  );
}

function LinkRow({ label, refs, onOpen }: { label: string; refs: NoteRef[]; onOpen: (p: string) => void }) {
  if (!refs.length) return null;
  return (
    <p className="linkrow">
      <span className="linkrow-label">{label}</span>{" "}
      {refs.map((r, i) => (
        <span key={r.name}>
          {i > 0 ? ", " : null}
          {r.path ? (
            <button className="wikilink" onClick={() => onOpen(r.path!)}>{r.name}</button>
          ) : (
            <span className="wikilink wikilink-missing" title="Not in the vault yet">{r.name}</span>
          )}
        </span>
      ))}
    </p>
  );
}

function NotePreview({ path, onOpen, onClose }: { path: string; onOpen: (p: string) => void; onClose: () => void }) {
  const [note, setNote] = useState<Note | null>(null);
  const [error, setError] = useState<string | null>(null);
  const panel = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let live = true; // ignore a slow reply for a note we've already moved past
    setNote(null);
    setError(null);
    getNote(path).then(
      (n) => live && setNote(n),
      (e: Error) => live && setError(e.message),
    );
    panel.current?.scrollTo(0, 0);
    return () => {
      live = false;
    };
  }, [path]);

  useEffect(() => {
    panel.current?.focus();
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const byName = new Map((note?.links ?? []).map((l) => [l.name.toLowerCase(), l.path]));
  const tags = note ? asList(note.frontmatter.tags) : [];

  return (
    <div className="scrim" onClick={onClose}>
      <div
        className="drawer"
        role="dialog"
        aria-modal="true"
        aria-label={note?.title ?? "Note preview"}
        tabIndex={-1}
        ref={panel}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="drawer-bar">
          <span className="source-path">{path}</span>
          <button className="text-button" onClick={onClose}>Close</button>
        </div>
        {error ? <p className="error">{error}</p> : null}
        {!note && !error ? <p className="muted pulse">Opening note…</p> : null}
        {note ? (
          <>
            <h2 className="drawer-title">{note.title}</h2>
            {tags.length ? (
              <p className="tags">{tags.map((t) => <span key={t} className="tag">#{t}</span>)}</p>
            ) : null}
            <div className="drawer-actions">
              <a className="button" href={note.obsidian_url}>Open in Obsidian</a>
            </div>
            <div className="note-body">
              <Markdown
                text={stripTitle(note.body, note.title)}
                onLink={(name) => {
                  const target = byName.get(name.toLowerCase());
                  if (target) onOpen(target);
                }}
              />
            </div>
            <div className="drawer-links">
              <LinkRow label="Links to" refs={note.links} onOpen={onOpen} />
              <LinkRow label="Linked from" refs={note.backlinks} onOpen={onOpen} />
              {!note.backlinks.length ? <p className="muted small">No other note links here yet.</p> : null}
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
}

function stripTitle(body: string, title: string) {
  const lines = body.replace(/^\s+/, "").split("\n");
  if (lines[0]?.replace(/^#\s+/, "").trim() === title) return lines.slice(1).join("\n");
  return body;
}

function asList(value: unknown): string[] {
  if (Array.isArray(value)) return value.map(String);
  if (typeof value === "string" && value) return [value];
  return [];
}

function fmtMs(ms?: number) {
  if (ms === undefined || ms === null) return "…";
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`;
}

function ago(iso: string) {
  const seconds = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
  if (seconds < 60) return "Just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 48) return `${hours} h ago`;
  return new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short" });
}
