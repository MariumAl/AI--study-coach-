"""
GRAPH — this is the only file that actually touches the LangGraph API.

MULTI-AGENT, added in this stage:

    quiz -> ask_answers -> evaluate -> supervisor -> (retest | review_notes | done)

evaluate no longer decides the loop-back itself. It hands off to a
SUPERVISOR node, which is a genuine second decision-maker: it looks at the
same state (weak_topics, retry_round) and chooses between delegating to
the Quiz Writer again ("retest") or the Notes Reviewer specialist
("review_notes") first. That choice is an LLM call (supervisor_llm in
nodes.py) wrapped in a deterministic hard-stop check — the retry cap is
enforced in plain Python BEFORE the supervisor is even asked, so the
safety limit never depends on the model's judgment.

review_notes always loops back to quiz once it's done — it's a detour
before more testing, not an endpoint.

Everything from earlier stages (the CYCLE, the CHECKPOINTER) still
applies exactly as before — this stage adds a second kind of decision
point (a coordinator agent), not a new mechanic.
"""

import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from state import CoachState
from nodes import (
    read_pdf, summarize, make_notes, make_flashcards,
    make_quiz, ask_answers, evaluate, supervisor, review_notes,
)


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


def route_after_supervisor(state: CoachState) -> str:
    """Just look up what the supervisor already decided — the JUDGMENT
    happened inside the supervisor node itself (LLM call + hard-stop
    check). This router is deliberately dumb; it only reads the answer."""
    action = state.get("next_action", "done")
    return {"retest": "quiz", "review_notes": "review_notes"}.get(action, "end")


graph_builder = StateGraph(CoachState)

graph_builder.add_node("read_pdf", read_pdf)
graph_builder.add_node("summarize", summarize)
graph_builder.add_node("make_notes", make_notes)
graph_builder.add_node("flashcards", make_flashcards)
graph_builder.add_node("quiz", make_quiz)
graph_builder.add_node("ask_answers", ask_answers)
graph_builder.add_node("evaluate", evaluate)
graph_builder.add_node("supervisor", supervisor)
graph_builder.add_node("review_notes", review_notes)

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

# quiz -> ask_answers -> evaluate -> supervisor is always a fixed sequence...
graph_builder.add_edge("quiz", "ask_answers")
graph_builder.add_edge("ask_answers", "evaluate")
graph_builder.add_edge("evaluate", "supervisor")

# ...but the supervisor's next step is a genuine multi-way decision.
graph_builder.add_conditional_edges(
    "supervisor",
    route_after_supervisor,
    {"quiz": "quiz", "review_notes": "review_notes", "end": END},
)

# review_notes is always a detour back into another quiz round.
graph_builder.add_edge("review_notes", "quiz")

conn = sqlite3.connect("study_coach_memory.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph = graph_builder.compile(checkpointer=checkpointer)
