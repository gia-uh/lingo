"""Examples package.

Sets a dummy API key when none is configured so that example modules can be
safely imported in test environments without starting a real LLM connection.
"""

import os

os.environ.setdefault("API_KEY", "dummy-for-import")
