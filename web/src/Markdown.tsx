import { Fragment, type ReactNode } from "react";

/**
 * A small Markdown renderer that builds React nodes (no innerHTML), enough
 * for model answers and note previews: headings, paragraphs, lists, code,
 * bold, italics, inline code, [1] citations and [[wikilinks]].
 */
type Props = {
  text: string;
  onCite?: (n: number) => void;
  onLink?: (name: string) => void;
  citable?: Set<number>;
};

const INLINE = /(\*\*[^*]+\*\*|`[^`]+`|\[\[[^\]]+\]\]|\[\d+(?:,\s*\d+)*\]|\*[^*\s][^*]*\*|_[^_\s][^_]*_)/g;

function inline(text: string, props: Props, keyBase: string): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  let i = 0;
  for (const match of text.matchAll(INLINE)) {
    const token = match[0];
    const start = match.index ?? 0;
    if (start > last) out.push(text.slice(last, start));
    const key = `${keyBase}-${i++}`;
    if (token.startsWith("**")) {
      out.push(<strong key={key}>{token.slice(2, -2)}</strong>);
    } else if (token.startsWith("`")) {
      out.push(<code key={key}>{token.slice(1, -1)}</code>);
    } else if (token.startsWith("[[")) {
      const inner = token.slice(2, -2);
      const [target, alias] = inner.split("|");
      const name = target.split("#")[0].split("/").pop()!.trim();
      out.push(
        props.onLink ? (
          <button key={key} className="wikilink" onClick={() => props.onLink!(name)}>
            {(alias ?? target).trim()}
          </button>
        ) : (
          <span key={key} className="wikilink">{(alias ?? target).trim()}</span>
        ),
      );
    } else if (token.startsWith("[")) {
      const numbers = token.slice(1, -1).split(",").map((n) => Number(n.trim()));
      out.push(
        <span key={key} className="cites">
          {numbers.map((n) =>
            props.onCite && (!props.citable || props.citable.has(n)) ? (
              <button key={n} className="cite" onClick={() => props.onCite!(n)} aria-label={`Source ${n}`}>
                {n}
              </button>
            ) : (
              <sup key={n} className="cite cite-plain">{n}</sup>
            ),
          )}
        </span>,
      );
    } else {
      out.push(<em key={key}>{token.slice(1, -1)}</em>);
    }
    last = start + token.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export function Markdown(props: Props) {
  const lines = props.text.replace(/\r\n/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  let paragraph: string[] = [];
  let list: { ordered: boolean; items: string[] } | null = null;
  let code: string[] | null = null;

  const flushParagraph = () => {
    if (paragraph.length) {
      const k = `p${blocks.length}`;
      blocks.push(<p key={k}>{inline(paragraph.join(" "), props, k)}</p>);
      paragraph = [];
    }
  };
  const flushList = () => {
    if (list) {
      const k = `l${blocks.length}`;
      const items = list.items.map((item, i) => <li key={i}>{inline(item, props, `${k}-${i}`)}</li>);
      blocks.push(list.ordered ? <ol key={k}>{items}</ol> : <ul key={k}>{items}</ul>);
      list = null;
    }
  };

  for (const line of lines) {
    if (code) {
      if (line.trim().startsWith("```")) {
        blocks.push(<pre key={`c${blocks.length}`}><code>{code.join("\n")}</code></pre>);
        code = null;
      } else code.push(line);
      continue;
    }
    if (line.trim().startsWith("```")) {
      flushParagraph();
      flushList();
      code = [];
      continue;
    }
    const heading = /^(#{1,6})\s+(.*)$/.exec(line);
    const bullet = /^\s*[-*+]\s+(?:\[[ x]\]\s+)?(.*)$/.exec(line);
    const numbered = /^\s*\d+[.)]\s+(.*)$/.exec(line);
    if (heading) {
      flushParagraph();
      flushList();
      const level = Math.min(heading[1].length + 1, 4);
      const Tag = `h${level}` as "h2" | "h3" | "h4";
      const k = `h${blocks.length}`;
      blocks.push(<Tag key={k}>{inline(heading[2], props, k)}</Tag>);
    } else if (bullet || numbered) {
      flushParagraph();
      const ordered = !bullet;
      if (list && list.ordered !== ordered) flushList();
      list = list ?? { ordered, items: [] };
      list.items.push((bullet ?? numbered)![1]);
    } else if (!line.trim()) {
      flushParagraph();
      flushList();
    } else {
      flushList();
      paragraph.push(line.trim());
    }
  }
  if (code) blocks.push(<pre key="c-open"><code>{(code as string[]).join("\n")}</code></pre>);
  flushParagraph();
  flushList();
  return <Fragment>{blocks}</Fragment>;
}
