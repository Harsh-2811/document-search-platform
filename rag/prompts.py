"""D3 — load prompt templates from disk so they can be edited without code.

The QA prompt and the CrewAI agent/task prompts used to live as string
literals inside `rag/retrieval.py` and `rag/agents.py`. Moving them out:

- lets non-engineers edit tone, rules, or output format without a redeploy
- makes the prompts diff-able in PR review
- keeps the Python code focused on plumbing

Both loaders are process-cached so a chat request doesn't re-read the
files on every call. Files are loaded once at first use and never watched
for changes mid-process — edits land on the next process restart, which is
the same contract as the rest of the Django app.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache(maxsize=1)
def load_qa_prompt() -> str:
    """Read the plain-text QA template used by the LlamaIndex engine."""
    return (_PROMPTS_DIR / "qa_prompt.txt").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def load_agents_yaml() -> dict:
    """Read the CrewAI agent/task definitions.

    Returned as the raw dict produced by PyYAML. The caller picks out the
    fields it needs (role, goal, backstory, etc.) — keeping the loader
    dumb means the YAML schema can grow without touching this module.
    """
    with (_PROMPTS_DIR / "agents.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)
