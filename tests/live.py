"""Run the tests that call the real Ollama on pc: python3 -m tests.live"""

import os
import sys
import unittest

os.environ["RAG_LIVE_OLLAMA"] = "1"
suite = unittest.defaultTestLoader.loadTestsFromName("tests.test_providers")
sys.exit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
