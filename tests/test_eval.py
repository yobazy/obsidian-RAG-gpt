import json
import unittest
from pathlib import Path

from eval import score_sources, summarize
from rag.notes import note_name
from rag.pipeline import markdown_files
from tests.fakes import FIXTURE_VAULT

QUESTIONS = Path(__file__).parent / "fixtures" / "questions.json"


class QuestionSetTest(unittest.TestCase):
    def setUp(self):
        self.questions = json.loads(QUESTIONS.read_text())

    def test_every_expected_source_is_a_fixture_note(self):
        names = {note_name(p) for p in markdown_files(FIXTURE_VAULT)}
        for item in self.questions:
            for source in item["expected_sources"]:
                self.assertIn(source, names, item["question"])

    def test_set_is_big_enough_and_covers_the_hard_cases(self):
        self.assertGreaterEqual(len(self.questions), 40)
        kinds = [item["type"] for item in self.questions]
        self.assertGreaterEqual(kinds.count("multi-hop"), 5)
        self.assertGreaterEqual(kinds.count("backlink"), 5)
        for item in self.questions:
            if item["type"] == "abstain":
                self.assertEqual(item["expected_sources"], [])
            else:
                self.assertTrue(item["expected_sources"], item["question"])
                self.assertTrue(item["expected_keywords"], item["question"])


class ScoreSourcesTest(unittest.TestCase):
    def test_recall_rank_and_hit(self):
        scores = score_sources(["B", "C"], ["A", "B", "B", "D", "C"])
        self.assertEqual(scores["recall"], 1.0)
        self.assertEqual(scores["recall3"], 0.5)  # unique order is A, B, D, C
        self.assertEqual(scores["rr"], 0.5)
        self.assertFalse(scores["hit1"])

    def test_case_insensitive_and_missing(self):
        scores = score_sources(["weekend loaf"], ["Weekend Loaf"])
        self.assertTrue(scores["hit1"])
        self.assertEqual(score_sources(["X"], [])["rr"], 0.0)

    def test_summary_keeps_abstain_cases_apart(self):
        results = [
            {"type": "fact", "source_recall": 1.0, "source_recall3": 1.0, "source_rr": 1.0,
             "source_hit1": True, "retrieval_score": 1.0, "abstained": None, "chunk_count": 2,
             "retrieval_ms": 5},
            {"type": "abstain", "source_recall": None, "source_recall3": None, "source_rr": None,
             "source_hit1": None, "retrieval_score": 0.0, "abstained": True, "chunk_count": 0,
             "retrieval_ms": 3},
        ]
        summary = summarize(results, retrieval_only=True)
        self.assertEqual(summary["source_recall"], 1.0)
        self.assertEqual(summary["keyword_retrieval"], 1.0)
        self.assertEqual(summary["abstain_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
