"""
STATE — the shared object every node reads from and writes to.

We use a TypedDict here (not a Pydantic model) because LangGraph's
StateGraph natively merges plain dicts into a TypedDict on every node
return. Pydantic works too (Stage 3+), but TypedDict is the simplest
way to *see* what's happening while you're learning.

Every field below is something a LATER node will need. We're only
filling in file_path, raw_text, and summary in Stage 1 — the rest
exist now so you can see the whole shape of where this is going,
but they'll stay empty until we build the nodes that fill them.
"""

from typing import TypedDict, List, Literal


class CoachState(TypedDict):
    file_path: str          # input: path to the lecture PDF
    mode: Literal["summary", "notes", "both"]  # input: what to generate this run
    raw_text: str            # filled by read_pdf node
    summary: str             # filled by summarize node (empty if mode="notes")
    notes: str               # filled by make_notes node (empty if mode="summary")
    flashcards: List[dict]   # Stage 2
    quiz: List[dict]         # Stage 2
    weak_topics: List[str]   # Stage 3 — this is the "memory" part
