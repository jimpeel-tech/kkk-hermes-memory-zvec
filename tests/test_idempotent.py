"""Isolated tests; never access the user's Hermes/Zvec production paths."""
import importlib.util
import sys
import tempfile
import threading
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent.parent
agent = types.ModuleType("agent")
agent.__path__ = []
provider_module = types.ModuleType("agent.memory_provider")
provider_module.MemoryProvider = type("MemoryProvider", (), {})
tools = types.ModuleType("tools")
tools.__path__ = []
registry = types.ModuleType("tools.registry")
registry.tool_error = lambda message: message
sys.modules["agent"] = agent
sys.modules["agent.memory_provider"] = provider_module
sys.modules["tools"] = tools
sys.modules["tools.registry"] = registry
spec = importlib.util.spec_from_file_location("memory_zvec_under_test", root / "__init__.py")
plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin)


class Status:
    def __init__(self, code=None, message=""):
        self._code, self._message = code, message
    def ok(self): return self._code is None
    def code(self): return self._code
    def message(self): return self._message


class Doc:
    def __init__(self, id, vectors, fields):
        self.id, self.vectors, self.fields = id, vectors, fields


class FakeZvec:
    Doc = Doc
    StatusCode = types.SimpleNamespace(ALREADY_EXISTS="ALREADY_EXISTS")


class Collection:
    def __init__(self):
        self.docs, self.fail_next, self.flushes = {}, False, 0
    def fetch(self, ids, *, include_vector=True):
        ids = [ids] if isinstance(ids, str) else ids
        return {i: self.docs[i] for i in ids if i in self.docs}
    def insert(self, doc):
        if self.fail_next:
            self.fail_next = False
            return Status("UNAVAILABLE", "simulated failure")
        if doc.id in self.docs:
            return Status("ALREADY_EXISTS", "already exists")
        self.docs[doc.id] = doc
        return Status()
    def flush(self): self.flushes += 1
    def optimize(self): pass


class WriteTests(unittest.TestCase):
    def setUp(self):
        self.coll = Collection()
        p = object.__new__(plugin.ZvecMemoryProvider)
        self.p = p
        p._config = {"embedding_model": "nomic-embed-text", "min_content_len": 50,
                     "enable_hnsw_optimize": False}
        p._insert_lock = threading.Lock()
        p._embedding_lock = threading.Lock()
        p._writes_since_optimize = 0
        p._session_id = "session-a"
        p._coll = self.coll
        p._read_only = False
        p._ensure_open = lambda: None
        p._prefetch_cache = {}
        p._prefetch_ttl = 30
        self.num_embeds = 0
        self.orig_zvec, self.orig_embed = plugin._get_zvec, plugin._ollama_embed_single
        plugin._get_zvec = lambda: FakeZvec
        def embed(*args):
            self.num_embeds += 1
            return np.ones(3, dtype=np.float32)
        plugin._ollama_embed_single = embed
        self.user = "Project Indigo Heron database question"
        self.answer = "PostgreSQL and Redis with three Kubernetes nodes"

    def tearDown(self):
        plugin._drain_pending_writes(timeout=5)
        plugin._get_zvec, plugin._ollama_embed_single = self.orig_zvec, self.orig_embed

    def store(self, *, sid="session-a", role="turn", user=None, answer=None):
        return self.p._write_turn_once(
            user_content=self.user if user is None else user,
            assistant_content=self.answer if answer is None else answer,
            sid=sid, role=role, ts=1000, meta="{}", coll=self.coll)

    def test_idempotent_skips_duplicate_embedding(self):
        self.assertTrue(self.store())
        self.assertFalse(self.store(role="session_end"))
        self.assertEqual((len(self.coll.docs), self.num_embeds), (1, 1))

    def test_normalization_and_different_sessions(self):
        self.assertTrue(self.store(user="  "+self.user+"\r\n", answer=self.answer+"\n"))
        self.assertFalse(self.store())
        self.assertTrue(self.store(sid="session-b"))
        self.assertEqual(len(self.coll.docs), 2)

    def test_session_end_recovers_and_is_repeatable(self):
        messages = [{"role": "user", "content": self.user},
                    {"role": "assistant", "content": self.answer}]
        self.p.on_session_end(messages)
        plugin._drain_pending_writes(timeout=5)
        self.assertEqual(len(self.coll.docs), 1)
        self.assertEqual(next(iter(self.coll.docs.values())).fields["role"], "session_end")
        self.p.on_session_end(messages)
        plugin._drain_pending_writes(timeout=5)
        self.assertEqual(len(self.coll.docs), 1)

    def test_turn_and_session_end_share_document(self):
        self.p.sync_turn("  "+self.user, self.answer+"\n", session_id="session-a")
        plugin._drain_pending_writes(timeout=5)
        self.assertEqual(len(self.coll.docs), 1)
        self.p.on_session_end([{"role": "user", "content": self.user},
                               {"role": "assistant", "content": self.answer}])
        plugin._drain_pending_writes(timeout=5)
        self.assertEqual(len(self.coll.docs), 1)

    def test_concurrency(self):
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: self.store(), range(20)))
        self.assertEqual((results.count(True), len(self.coll.docs), self.num_embeds), (1, 1, 1))

    def test_failed_insert_fails_loudly_then_recovers(self):
        self.coll.fail_next = True
        with self.assertRaisesRegex(RuntimeError, "simulated failure"):
            self.store()
        self.assertTrue(self.store())
        self.assertEqual(len(self.coll.docs), 1)

    def test_prefetch_ignores_session_filter(self):
        self.p._config["prefetch_mode"] = "vector"
        seen, event = [], threading.Event()
        def fake_search(query, top_k, session_id):
            seen.append(session_id)
            event.set()
            return []
        self.p._do_search = fake_search
        self.p.prefetch("indigo heron", session_id="different")
        self.p.queue_prefetch("other query", session_id="different")
        self.assertTrue(event.wait(timeout=3))
        self.assertEqual(seen[0], "")
        # Both prefetch implementations are intentionally global:
        self.assertEqual(plugin._turn_doc_id("s", self.user, self.answer),
                         plugin._turn_doc_id("s", self.user, self.answer))


class SelectiveFactsTests(unittest.TestCase):
    """Exercise opt-in behavior using fake Ollama; no network or prod data."""

    def setUp(self):
        WriteTests.setUp(self)
        self.p._config["memory_write_mode"] = "facts"
        self.p._config["extraction_model"] = "qwen3.5:9b"
        self.orig_extract = plugin._ollama_extract_fact
        self.extractions = 0
        def extract(user, *args, **kwargs):
            self.extractions += 1
            return "Project Silver Badger uses CockroachDB, RabbitMQ and five worker nodes."
        plugin._ollama_extract_fact = extract

    def tearDown(self):
        plugin._ollama_extract_fact = self.orig_extract
        WriteTests.tearDown(self)

    store = WriteTests.store

    def test_facts_mode_stores_compact_fact_not_answer(self):
        self.assertTrue(self.store())
        row = next(iter(self.coll.docs.values()))
        self.assertEqual(row.fields["role"], "fact")
        self.assertIn("Silver Badger", row.fields["content"])
        self.assertNotIn("[assistant]", row.fields["content"])

    def test_repeated_session_end_does_not_regenerate(self):
        self.assertTrue(self.store(role="turn"))
        self.assertFalse(self.store(role="session_end"))
        self.assertEqual((len(self.coll.docs), self.extractions), (1, 1))

    def test_different_assistant_content_is_same_user_fact(self):
        self.assertTrue(self.store())
        self.assertFalse(self.store(answer="Different assistant reply"))
        self.assertEqual((len(self.coll.docs), self.extractions), (1, 1))

    def test_empty_extraction_skips_write(self):
        plugin._ollama_extract_fact = lambda *a, **k: ""
        self.assertFalse(self.store())
        self.assertEqual(len(self.coll.docs), 0)

    def test_failed_extraction_logs_and_does_not_write_raw_turn(self):
        def broken(*args, **kwargs):
            raise RuntimeError("Ollama unavailable")
        plugin._ollama_extract_fact = broken
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            self.store()
        self.assertEqual(len(self.coll.docs), 0)

    def test_selective_extractor_json_contract(self):
        class FakeResponse:
            def raise_for_status(self): return None
            def json(self): return {"response": '{"memory": "User prefers Fish shell."}'}
        old_post = plugin.requests.post
        previous_extractor = plugin._ollama_extract_fact
        plugin._ollama_extract_fact = self.orig_extract
        seen = {}
        def fake_post(url, **kwargs):
            seen["url"] = url
            seen["payload"] = kwargs["json"]
            return FakeResponse()
        try:
            plugin.requests.post = fake_post
            fact = plugin._ollama_extract_fact(
                "I prefer Fish for commands", "http://localhost:11434", "qwen3.5:9b")
            self.assertEqual(fact, "User prefers Fish shell.")
            self.assertTrue(seen["url"].endswith("/api/generate"))
            self.assertEqual(seen["payload"]["format"], "json")
            self.assertFalse(seen["payload"]["stream"])
        finally:
            plugin.requests.post = old_post
            plugin._ollama_extract_fact = previous_extractor

    def test_session_end_selective_extraction_is_bounded(self):
        self.p._config["facts_session_end_max_turns"] = 2
        messages = []
        for index in range(7):
            messages.extend([
                {"role": "user", "content": f"I use a tool on Project {index} and decided to keep it."},
                {"role": "assistant", "content": "Understood, noted."},
            ])
        self.p.on_session_end(messages)
        plugin._drain_pending_writes(timeout=5)
        self.assertEqual((len(self.coll.docs), self.extractions), (2, 2))

    def test_prefetch_hybrid_and_vector_fallback(self):
        self.p._config["prefetch_top_k"] = 5
        seen = []
        def hybrid(query, top_k, session_id):
            seen.append(("hybrid", top_k, session_id))
            return []
        def vector(query, top_k, session_id):
            seen.append(("vector", top_k, session_id))
            return [{"content": "Silver Badger", "score": 0.91}]
        self.p._do_hybrid_search = hybrid
        self.p._do_search = vector
        result = self.p.prefetch("Silver Badger?", session_id="new-session")
        self.assertIn("Silver Badger", result)
        self.assertEqual(seen, [("hybrid", 50, ""), ("vector", 50, "")])

    def test_prefetch_promotes_fact_below_top_five(self):
        self.p._config["prefetch_top_k"] = 5
        candidates = [
            {"content": f"Old generic troubleshooting transcript {i}", "role": "turn",
             "score": 0.032 - i * 0.001}
            for i in range(8)
        ]
        candidates.append({
            "content": "For Hermes, prefers Markdown tables then short PowerShell commands.",
            "role": "fact", "score": 0.019,
        })
        self.p._do_hybrid_search = lambda q, top_k, session_id: candidates
        results = self.p._prefetch_results(
            "When helping me troubleshoot Hermes, how should you format your answers?")
        self.assertEqual(len(results), 5)
        self.assertEqual(results[0]["role"], "fact")
        self.assertIn("Markdown tables", results[0]["content"])

    def test_prefetch_deduplicates_old_turns(self):
        results = [
            {"content": "User discussed a disk outage", "role": "turn"},
            {"content": "User discussed a disk outage", "role": "session_end"},
            {"content": "Prefers Fish syntax", "role": "fact"},
            {"content": "Unrelated detail", "role": "turn"},
        ]
        selected = self.p._rank_prefetch_candidates(results, 4)
        self.assertEqual([r["content"] for r in selected],
                         ["Prefers Fish syntax", "User discussed a disk outage", "Unrelated detail"])


class NativeApiTests(unittest.TestCase):
    def test_native_idempotent_write_with_full_sha256_id(self):
        """Use real Zvec 0.6.0 to write the 72-char IDs used by the fork."""
        try:
            import zvec
        except ImportError:
            self.skipTest("requires zvec==0.6.0")
        with tempfile.TemporaryDirectory() as tmp:
            schema = plugin._build_memories_schema(vector_dim=3, collection_name="test_memories")
            coll = zvec.create_and_open(str(Path(tmp) / "test_memories"), schema)
            try:
                p = object.__new__(plugin.ZvecMemoryProvider)
                p._config = {"embedding_model": "nomic-embed-text", "min_content_len": 50,
                             "enable_hnsw_optimize": False}
                p._insert_lock = threading.Lock()
                p._embedding_lock = threading.Lock()
                p._writes_since_optimize = 0
                original_embed = plugin._ollama_embed_single
                try:
                    plugin._ollama_embed_single = lambda *args: np.ones(3, dtype=np.float32)
                    user = "Project Copper Lantern uses MariaDB and NATS"
                    assistant = "Confirmed four Kubernetes worker nodes"
                    expected_id = plugin._turn_doc_id("session-test", user, assistant)
                    self.assertEqual(len(expected_id), 64)
                    self.assertTrue(all(c in "0123456789abcdef" for c in expected_id))
                    self.assertTrue(p._write_turn_once(
                        user_content=user, assistant_content=assistant,
                        sid="session-test", role="turn", ts=1000.0, meta="{}", coll=coll))
                    self.assertIn(expected_id, coll.fetch(ids=expected_id, include_vector=False))
                    self.assertFalse(p._write_turn_once(
                        user_content=user, assistant_content=assistant,
                        sid="session-test", role="session_end", ts=1000.0,
                        meta="{}", coll=coll))
                finally:
                    plugin._ollama_embed_single = original_embed
            finally:
                del coll

    def test_native_selective_fact_write(self):
        """Real Zvec 0.6.0 insert, duplicate check, and fact role."""
        try:
            import zvec
        except ImportError:
            self.skipTest("requires zvec==0.6.0")
        with tempfile.TemporaryDirectory() as tmp:
            coll = zvec.create_and_open(
                str(Path(tmp) / "facts"),
                plugin._build_memories_schema(vector_dim=3, collection_name="facts"))
            try:
                p = object.__new__(plugin.ZvecMemoryProvider)
                p._config = {"memory_write_mode": "facts", "min_content_len": 50,
                             "enable_hnsw_optimize": False}
                p._insert_lock, p._embedding_lock = threading.Lock(), threading.Lock()
                p._writes_since_optimize = 0
                old_embed, old_extract = plugin._ollama_embed_single, plugin._ollama_extract_fact
                try:
                    plugin._ollama_embed_single = lambda *args: np.ones(3, dtype=np.float32)
                    plugin._ollama_extract_fact = lambda *args, **kw: (
                        "Silver Badger runs CockroachDB and RabbitMQ on five Kubernetes nodes.")
                    u = "Silver Badger uses CockroachDB and RabbitMQ on five Kubernetes nodes."
                    self.assertTrue(p._write_turn_once(
                        user_content=u, assistant_content="Sure.",
                        sid="test-session", ts=1000.0, meta="{}", role="turn", coll=coll))
                    self.assertFalse(p._write_turn_once(
                        user_content=u, assistant_content="Different reply",
                        sid="test-session", ts=1001.0, meta="{}", role="session_end", coll=coll))
                    key = plugin._fact_doc_id("test-session", u)
                    self.assertEqual(len(key), 64)
                    self.assertIn(key, coll.fetch(ids=key, include_vector=False))
                    self.assertEqual(coll.fetch(ids=key)[key].fields["role"], "fact")
                finally:
                    plugin._ollama_embed_single, plugin._ollama_extract_fact = old_embed, old_extract
            finally:
                del coll

    def test_zvec_06_native_status_and_fetch(self):
        try:
            import zvec
        except ImportError:
            self.skipTest("requires zvec==0.6.0")
        with tempfile.TemporaryDirectory() as tmp:
            schema = plugin._build_memories_schema(vector_dim=3, collection_name="test_memories")
            coll = zvec.create_and_open(str(Path(tmp) / "test_memories"), schema)
            try:
                doc = zvec.Doc(
                    id="test-id",
                    vectors={"vector": [0.1, 0.2, 0.3]},
                    fields={"content": "a real Zvec test", "role": "turn",
                            "session_id": "s", "created_at": 1.0, "metadata": "{}"},
                )
                self.assertTrue(coll.insert(doc).ok())
                self.assertIn("test-id", coll.fetch(ids="test-id", include_vector=False))
                status = coll.insert(doc)
                self.assertFalse(status.ok())
                self.assertEqual(status.code(), zvec.StatusCode.ALREADY_EXISTS)
            finally:
                del coll

if __name__ == "__main__":
    unittest.main()
