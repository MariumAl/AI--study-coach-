"""
GRAPH — this is the only file that actually touches the LangGraph API.

StateGraph(CoachState)   -> "build a graph whose shared state has this shape"
.add_node(name, fn)      -> register a function under a name
.add_edge(A, B)          -> after A finishes, ALWAYS run B
.add_conditional_edges(A, router_fn, path_map)
                         -> after A finishes, call router_fn(state); whatever
                            string it returns gets looked up in path_map to
                            decide which node runs next. This is how a graph
                            makes a DECISION instead of following a fixed order.
START / END              -> special markers for "graph entry" and "graph exit"
.compile()               -> turns the definition into a runnable object
"""

from langgraph.graph import StateGraph, START, END
from state import CoachState
from nodes import read_pdf, summarize, make_notes


def route_after_read(state: CoachState) -> str:
    """Decide what to generate first, based on the mode the caller asked for."""
    if state["mode"] == "notes":
        return "make_notes"
    return "summarize"  # mode == "summary" or "both" both start with summarize


def route_after_summarize(state: CoachState) -> str:
    """After the summary, only continue to notes if the caller asked for both."""
    if state["mode"] == "both":
        return "make_notes"
    return "end"


graph_builder = StateGraph(CoachState)

graph_builder.add_node("read_pdf", read_pdf)
graph_builder.add_node("summarize", summarize)
graph_builder.add_node("make_notes", make_notes)

graph_builder.add_edge(START, "read_pdf")

# read_pdf is followed by a DECISION, not a fixed next step:
graph_builder.add_conditional_edges(
    "read_pdf",
    route_after_read,
    {"summarize": "summarize", "make_notes": "make_notes"},
)

# summarize is also followed by a decision: stop here, or continue to notes?
graph_builder.add_conditional_edges(
    "summarize",
    route_after_summarize,
    {"make_notes": "make_notes", "end": END},
)

graph_builder.add_edge("make_notes", END)

graph = graph_builder.compile()
