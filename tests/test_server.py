import json
import threading
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
        self.assertEqual(body["index"]["notes"], 8)
        self.assertEqual(body["last_ingest"]["files_seen"], 9)
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
        self.assertEqual(body["backlinks"], [{"name": "Sourdough Starter", "path": "Kitchen/Sourdough Starter.md"}])

    def test_unresolved_links_have_no_path(self):
        _, body = get(self.base, "/notes/" + quote("Coast Path Ride.md"))
        self.assertEqual(body["links"], [{"name": "Lighthouse App", "path": "Projects/Lighthouse App.md"}])

    def test_note_paths_cannot_escape_the_vault(self):
        for path in ["../../../README.md", quote("../test_notes.py", safe=""), ".obsidian/Ignored.md",
                     "Missing.md", quote("/etc/passwd", safe="")]:
            code, _ = http_error(lambda: get(self.base, "/notes/" + path))
            self.assertEqual(code, 404, path)


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
        self.assertEqual(body["index"]["notes"], 8)


if __name__ == "__main__":
    unittest.main()
