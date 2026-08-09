"""API key access. Keys live in the environment or .env (gitignored) —
never in code, never committed. Missing keys fail loudly at the point
of use, not silently at import."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def get_api_key(name: str) -> str | None:
    return os.environ.get(name)
