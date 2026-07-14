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

Step = Literal["summary", "notes", "flashcards", "quiz"]


class CoachState(TypedDict):
    file_path: str          # input: path to the lecture PDF
    steps: List[Step]        # input: which outputs to generate this run, any combo
    raw_text: str            # filled by read_pdf node
    summary: str             # filled by summarize node (only if "summary" in steps)
    notes: str               # filled by make_notes node (only if "notes" in steps)
    flashcards: List[dict]   # filled by flashcards node (only if "flashcards" in steps)
    quiz: List[dict]         # filled by quiz node (only if "quiz" in steps)
    answers: List[str]       # filled by ask_answers node — student's typed answers
    weak_topics: List[str]   # filled by evaluate node — topics the student got wrong
    retry_round: int         # how many times we've looped back for more practice
    last_feedback: List[dict]  # filled by evaluate node — per-question grading detail,
                                # so any frontend can display it (not just server print())
    next_action: str            # filled by supervisor node — "retest" / "review_notes" / "done"
    supervisor_reasoning: str   # filled by supervisor node — why it chose that action
    notes_reviewed: bool        # whether review_notes has already run this session
                                 # (prevents the supervisor looping on that specialist forever)
    focused_review: str         # filled by review_notes node — targeted re-explanation
                                 # of weak topics only, shown before the next quiz round
