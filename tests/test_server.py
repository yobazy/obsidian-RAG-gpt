import http.client
import json
import os
import threading
import time
import unittest
import urllib.error
import urllib.request
from urllib.parse import quote

from rag.config import Settings
from rag.store import MemoryStore
from server import App, make_server
from tests.fakes import FIXTURE_VAULT, DownProvider, FakeProvider


def start(provider):
    settings = Settings(provider="local", store="memory", vault_path=str(FIXTURE_VAULT),
                        vault_name="baz-vault", min_similarity=0.1)
    app = App(settings, store=MemoryStore(), provider=FakeProvider())
    app.ingest()  # index with the working fake, then swap in the provider under test
    app.provider = provider
    server = make_server(app, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return app, server, f"http://127.0.0.1:{server.server_address[1]}"


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=10) as r:
        return r.status, json.load(r)


def post(base, path, payload):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=10)


def http_error(fn):
    try:
        fn()
    except urllib.error.HTTPError as e:
        return e.code, json.load(e)
    raise AssertionError("expected an HTTP error")


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app, cls.server, cls.base = start(FakeProvider(reply="Bramble gets fed weekly [1]."))

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_status(self):
        status, body = get(self.base, "/status")
        self.assertEqual(status, 200)
        self.assertEqual(body["provider"], "local")
        self.assertEqual(body["index"]["notes"], 24)
        self.assertEqual(body["last_ingest"]["files_seen"], 25)
        self.assertTrue(body["backend"]["reachable"])

    def test_ask_streams_sources_then_tokens(self):
        with post(self.base, "/ask", {"question": "how do I feed the sourdough starter Bramble"}) as r:
            self.assertEqual(r.headers["Content-Type"], "application/x-ndjson; charset=utf-8")
            events = [json.loads(line) for line in r if line.strip()]
        types = [e["type"] for e in events]
        self.assertEqual(types[0], "sources")
        self.assertEqual(types[-1], "done")
        self.assertIn("token", types)
        answer = "".join(e["text"] for e in events if e["type"] == "token")
        self.assertIn("Bramble", answer)

        top = events[0]["sources"][0]
        self.assertEqual(top["path"], "Kitchen/Sourdough Starter.md")
        self.assertEqual(top["n"], 1)
        self.assertIsInstance(top["score"], float)
        self.assertIn("heading", top)
        self.assertEqual(top["obsidian_url"],
                         "obsidian://open?vault=baz-vault&file=Kitchen%2FSourdough%20Starter")
        self.assertIn({"name": "Weekend Loaf", "path": "Kitchen/Weekend Loaf.md"}, top["links"])
        self.assertIn({"name": "Weekend Loaf", "path": "Kitchen/Weekend Loaf.md"}, top["backlinks"])

    def test_snippet_skips_tags_and_title(self):
        from server import snippet_text

        self.assertEqual(snippet_text("Tags: baking, recipe Weekend Loaf The recipe", "Weekend Loaf"), "The recipe")
        self.assertEqual(snippet_text("Garden Log Tomatoes Planted", "Garden Log"), "Tomatoes Planted")
        self.assertEqual(snippet_text("Mid-note text", "Garden Log"), "Mid-note text")

    def test_ask_needs_a_question(self):
        code, body = http_error(lambda: post(self.base, "/ask", {"question": "  "}))
        self.assertEqual(code, 400)
        self.assertIn("error", body)

    def test_search(self):
        status, body = get(self.base, "/search?q=" + quote("tide station code NB04"))
        self.assertEqual(status, 200)
        self.assertEqual(body["results"][0]["path"], "Projects/Tide Data Source.md")
        linked = [r for r in body["results"] if r["via_link"]]
        self.assertTrue(linked)
        self.assertTrue(all(r["score"] is None for r in linked))

    def test_search_needs_a_query(self):
        code, _ = http_error(lambda: get(self.base, "/search?q="))
        self.assertEqual(code, 400)

    def test_note_preview(self):
        status, body = get(self.base, "/notes/" + quote("Kitchen/Weekend Loaf.md"))
        self.assertEqual(status, 200)
        self.assertEqual(body["title"], "Weekend Loaf")
        self.assertEqual(body["frontmatter"]["tags"], ["baking", "recipe"])
        self.assertTrue(body["body"].lstrip().startswith("# Weekend Loaf"))
        self.assertTrue(body["indexed"])
        self.assertEqual(body["links"], [{"name": "Sourdough Starter", "path": "Kitchen/Sourdough Starter.md"}])
        self.assertEqual(body["backlinks"], [{"name": "Focaccia", "path": "Kitchen/Focaccia.md"},
                                             {"name": "Sourdough Starter", "path": "Kitchen/Sourdough Starter.md"}])

    def test_image_embeds_are_not_listed_as_links(self):
        _, body = get(self.base, "/notes/" + quote("Coast Path Ride.md"))
        self.assertEqual(body["links"], [{"name": "Lighthouse App", "path": "Projects/Lighthouse App.md"}])

    def test_unresolved_links_have_no_path(self):
        # No fixture note links to a missing note, so check the resolver directly.
        self.assertEqual(self.app.resolve_links(["Weekend Loaf", "Not Written Yet"]),
                         [{"name": "Weekend Loaf", "path": "Kitchen/Weekend Loaf.md"},
                          {"name": "Not Written Yet", "path": None}])

    def test_note_paths_cannot_escape_the_vault(self):
        # README.md exists two levels above tests/fixtures/vault, so these only
        # 404 because of the containment check, not because the file is missing.
        for path in ["../../../README.md", quote("../../../README.md", safe=""), ".obsidian/Ignored.md",
                     "Missing.md", quote("/etc/passwd", safe="")]:
            code, _ = http_error(lambda: get(self.base, "/notes/" + path))
            self.assertEqual(code, 404, path)

    def raw(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=10)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            r = conn.getresponse()
            return r.status, r.read()
        finally:
            conn.close()

    def test_other_host_names_are_refused(self):
        # A page that rebinds its own domain to 127.0.0.1 sends its domain as Host.
        status, _ = self.raw("GET", "/notes/" + quote("Kitchen/Weekend Loaf.md"),
                             headers={"Host": "attacker.example:8790"})
        self.assertEqual(status, 403)
        status, _ = self.raw("POST", "/ask", body=b'{"question": "starter"}',
                             headers={"Host": "attacker.example", "Content-Type": "application/json"})
        self.assertEqual(status, 403)
        status, _ = self.raw("GET", "/status", headers={"Host": "localhost:8790"})
        self.assertEqual(status, 200)

    def test_allowed_hosts_can_be_extended(self):
        os.environ["RAG_ALLOWED_HOSTS"] = "pc.example.ts.net"
        try:
            server = make_server(self.app, port=0)
        finally:
            del os.environ["RAG_ALLOWED_HOSTS"]
        self.assertIn("pc.example.ts.net", server.RequestHandlerClass.allowed_hosts)
        self.assertIn("127.0.0.1", server.RequestHandlerClass.allowed_hosts)
        server.server_close()
        wide = make_server(self.app, host="0.0.0.0", port=0)
        self.assertNotIn("0.0.0.0", wide.RequestHandlerClass.allowed_hosts)
        wide.server_close()

    def test_posts_must_be_json(self):
        # text/plain and form posts skip the CORS preflight, so any site could send them.
        for path in ("/ask", "/ingest"):
            status, _ = self.raw("POST", path, body=b'{"question": "starter"}',
                                 headers={"Content-Type": "text/plain"})
            self.assertEqual(status, 415, path)

    def test_ask_body_must_be_an_object(self):
        for body in (b"[]", b'"starter"', b"not json"):
            status, _ = self.raw("POST", "/ask", body=body, headers={"Content-Type": "application/json"})
            self.assertEqual(status, 400, body)
        with post(self.base, "/ask", {"question": "sourdough starter", "history": "oops"}) as r:
            events = [json.loads(line) for line in r if line.strip()]
        self.assertEqual(events[-1]["type"], "done")

    def test_search_k_is_bounded(self):
        for k in ("-1", "1000"):
            code, _ = http_error(lambda: get(self.base, f"/search?q=bread&k={k}"))
            self.assertEqual(code, 400, k)
        _, body = get(self.base, "/search?q=" + quote("sourdough starter") + "&k=1")
        self.assertEqual(len([r for r in body["results"] if not r["via_link"]]), 1)


class BrokenMidAnswerTest(unittest.TestCase):
    def test_unexpected_errors_end_the_stream_with_an_error_event(self):
        class Breaks(FakeProvider):
            def stream_chat(self, system, messages):
                yield "Half an "
                raise RuntimeError("boom")

        _, server, base = start(Breaks())
        try:
            with post(base, "/ask", {"question": "sourdough starter",
                                     "history": [{"role": "assistant", "content": ""}]}) as r:
                events = [json.loads(line) for line in r if line.strip()]
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["code"], "internal")


class BackendDownTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app, cls.server, cls.base = start(DownProvider())

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def test_ask_reports_a_readable_error(self):
        with post(self.base, "/ask", {"question": "anything"}) as r:
            events = [json.loads(line) for line in r if line.strip()]
        self.assertEqual(events[-1]["type"], "error")
        self.assertEqual(events[-1]["code"], "backend_unavailable")
        self.assertIn("Can't reach Ollama", events[-1]["message"])

    def test_search_is_503(self):
        code, body = http_error(lambda: get(self.base, "/search?q=bread"))
        self.assertEqual(code, 503)
        self.assertIn("Can't reach Ollama", body["error"])

    def test_status_still_works(self):
        _, body = get(self.base, "/status")
        self.assertFalse(body["backend"]["reachable"])
        self.assertEqual(body["index"]["notes"], 24)


class UnreachablePcFailsFastTest(unittest.TestCase):
    """When ping already says pc is down, don't wait out the embed timeout."""

    class Hangs(DownProvider):
        def embed(self, texts, kind):
            time.sleep(5)  # stands in for a connect to a pc that is switched off
            return super().embed(texts, kind)

    def setUp(self):
        self.app, self.server, self.base = start(self.Hangs())

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_ask_and_search_return_quickly(self):
        started = time.perf_counter()
        events = list(self.app.ask("anything"))
        self.assertEqual(events[-1]["code"], "backend_unavailable")
        self.assertIn("Can't reach Ollama", events[-1]["message"])
        code, _ = http_error(lambda: get(self.base, "/search?q=bread"))
        self.assertEqual(code, 503)
        self.assertLess(time.perf_counter() - started, 2)


if __name__ == "__main__":
    unittest.main()
