import unittest

from rag.notes import chunk_note, clean_markdown, extract_links, note_name, parse_frontmatter
from tests.fakes import FIXTURE_VAULT


class LinksTest(unittest.TestCase):
    def test_alias_heading_and_folder_are_normalised(self):
        text = "See [[Weekend Loaf]], [[Pantry Staples|pantry]], [[Lighthouse App#Decisions|x]] and [[Kitchen/Weekend Loaf]]."
        self.assertEqual(extract_links(text), ["Weekend Loaf", "Pantry Staples", "Lighthouse App"])

    def test_image_embeds_are_not_links(self):
        self.assertEqual(extract_links("![[ride-map.png]] and ![[Garden Log]]"), ["Garden Log"])

    def test_links_inside_code_blocks_are_ignored(self):
        self.assertEqual(extract_links("```\n[[Not A Link]]\n```\n[[Real]]"), ["Real"])


class CleanTest(unittest.TestCase):
    def test_link_text_survives_cleaning(self):
        cleaned = clean_markdown("## Notes\nFlour is in [[Pantry Staples|the pantry list]] and [[Garden Log]].")
        self.assertIn("the pantry list", cleaned)
        self.assertIn("Garden Log", cleaned)
        self.assertNotIn("[[", cleaned)
        self.assertNotIn("##", cleaned)


class ChunkTest(unittest.TestCase):
    def test_chunks_carry_their_heading(self):
        text = (FIXTURE_VAULT / "Kitchen" / "Weekend Loaf.md").read_text()
        chunks = chunk_note(text, chunk_size=30, overlap=5)
        self.assertGreater(len(chunks), 3)
        headings = {c.heading for c in chunks}
        self.assertIn("Method", headings)
        self.assertIn("Lessons", headings)
        lessons = [c for c in chunks if "78 percent" in c.content]
        self.assertTrue(lessons)

    def test_windows_overlap(self):
        text = " ".join(f"w{i}" for i in range(100))
        chunks = chunk_note(text, chunk_size=40, overlap=10)
        self.assertEqual([c.index for c in chunks], [0, 1, 2])
        self.assertEqual(chunks[1].content.split()[0], "w30")
        self.assertEqual(chunks[-1].content.split()[-1], "w99")

    def test_frontmatter_tags_kept_other_properties_dropped(self):
        text = (FIXTURE_VAULT / "Kitchen" / "Weekend Loaf.md").read_text()
        first = chunk_note(text)[0]
        self.assertTrue(first.content.startswith("Tags: baking, recipe Weekend Loaf"))
        self.assertNotIn("status", first.content)
        self.assertEqual(first.heading, "Weekend Loaf")

    def test_short_text_gives_one_chunk(self):
        self.assertEqual(len(chunk_note("just a few words here")), 1)


class FrontmatterTest(unittest.TestCase):
    def test_inline_and_block_lists(self):
        meta, body = parse_frontmatter((FIXTURE_VAULT / "Kitchen" / "Weekend Loaf.md").read_text())
        self.assertEqual(meta["tags"], ["baking", "recipe"])
        self.assertEqual(meta["status"], "settled")
        self.assertTrue(body.lstrip().startswith("# Weekend Loaf"))

        meta, _ = parse_frontmatter((FIXTURE_VAULT / "Kitchen" / "Pantry Staples.md").read_text())
        self.assertEqual(meta["tags"], ["kitchen", "shopping"])

    def test_no_frontmatter(self):
        self.assertEqual(parse_frontmatter("# Title\nbody"), ({}, "# Title\nbody"))

    def test_note_name(self):
        self.assertEqual(note_name("/v/Kitchen/Weekend Loaf.md"), "Weekend Loaf")


if __name__ == "__main__":
    unittest.main()
