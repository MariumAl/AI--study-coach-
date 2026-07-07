"""
GRAPH — this is the only file that actually touches the LangGraph API.

New in this stage: a CYCLE. Every edge so far only ever moved forward
(skip a stage, but never go back). Now:

    quiz -> ask_answers -> evaluate -> (loop back to quiz, OR end)

That loop-back is decided by route_after_evaluate(), based on state the
graph itself produced (weak_topics) — not something the user typed in.
This is the actual "agentic" mechanic: the system observes its own
output and decides whether to keep going.

MAX_RETRY_ROUNDS exists so a stubborn weak topic can't loop forever —
every loop in an agent needs a hard stop condition.

ALSO new: a CHECKPOINTER. Without one, state only exists for the duration
of a single .invoke() call — close the program and it's gone. A
checkpointer saves the state to disk (here, a local SQLite file) after
every node runs, keyed by a "thread_id" you choose. Invoke the graph again
later with the SAME thread_id, and any fields you don't explicitly
overwrite carry over from where they left off — that's how "quiz me
tomorrow" can remember today's weak topics without you re-supplying them.
"""

import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from state import CoachState
from nodes import (
    read_pdf, summarize, make_notes, make_flashcards,
    make_quiz, ask_answers, evaluate,
)

MAX_RETRY_ROUNDS = 2


def route_to_quiz_or_end(state: CoachState) -> str:
    if "quiz" in state["steps"]:
        return "quiz"
    return "end"


def route_to_flashcards_or_past(state: CoachState) -> str:
    if "flashcards" in state["steps"]:
        return "flashcards"
    return route_to_quiz_or_end(state)


def route_to_notes_or_past(state: CoachState) -> str:
    if "notes" in state["steps"]:
        return "make_notes"
    return route_to_flashcards_or_past(state)


def route_to_summary_or_past(state: CoachState) -> str:
    if "summary" in state["steps"]:
        return "summarize"
    return route_to_notes_or_past(state)


def route_after_evaluate(state: CoachState) -> str:
    """The loop-back decision. Keep practicing if: there ARE weak topics,
    AND we haven't hit the retry cap yet. Otherwise, stop."""
    still_weak = bool(state.get("weak_topics"))
    rounds_left = state.get("retry_round", 0) < MAX_RETRY_ROUNDS

    if still_weak and rounds_left:
        return "quiz"  # loop back — generate more questions on weak topics
    return "end"


graph_builder = StateGraph(CoachState)

graph_builder.add_node("read_pdf", read_pdf)
graph_builder.add_node("summarize", summarize)
graph_builder.add_node("make_notes", make_notes)
graph_builder.add_node("flashcards", make_flashcards)
graph_builder.add_node("quiz", make_quiz)
graph_builder.add_node("ask_answers", ask_answers)
graph_builder.add_node("evaluate", evaluate)

graph_builder.add_edge(START, "read_pdf")

graph_builder.add_conditional_edges(
    "read_pdf",
    route_to_summary_or_past,
    {"summarize": "summarize", "make_notes": "make_notes",
     "flashcards": "flashcards", "quiz": "quiz", "end": END},
)

graph_builder.add_conditional_edges(
    "summarize",
    route_to_notes_or_past,
    {"make_notes": "make_notes", "flashcards": "flashcards",
     "quiz": "quiz", "end": END},
)

graph_builder.add_conditional_edges(
    "make_notes",
    route_to_flashcards_or_past,
    {"flashcards": "flashcards", "quiz": "quiz", "end": END},
)

graph_builder.add_conditional_edges(
    "flashcards",
    route_to_quiz_or_end,
    {"quiz": "quiz", "end": END},
)

# quiz -> ask_answers -> evaluate is always a fixed sequence...
graph_builder.add_edge("quiz", "ask_answers")
graph_builder.add_edge("ask_answers", "evaluate")

# ...but evaluate's next step is a DECISION: loop back to quiz, or stop.
graph_builder.add_conditional_edges(
    "evaluate",
    route_after_evaluate,
    {"quiz": "quiz", "end": END},
)

conn = sqlite3.connect("study_coach_memory.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph = graph_builder.compile(checkpointer=checkpointer)
