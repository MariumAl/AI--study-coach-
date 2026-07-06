"""
NODES — plain functions. Each takes the current state and returns a dict
of the fields it wants to update. LangGraph merges that dict back into
the state before handing it to the next node.

Notice: nothing in here imports langgraph. Nodes don't need to know
they're part of a graph — that's the point. It keeps them simple and
independently testable (try calling read_pdf({...}) yourself in a
python shell, no graph required).
"""

import os
from pypdf import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from state import CoachState

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


def read_pdf(state: CoachState) -> dict:
    reader = PdfReader(state["file_path"])
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    print(f"[read_pdf] extracted {len(text)} characters from {len(reader.pages)} pages")
    return {"raw_text": text}


SUMMARY_SYSTEM_PROMPT = """You are a study coach creating exam-review summaries \
from lecture slides. Follow this exact structure:

## [Topic name]
- One tight paragraph (2-4 sentences) capturing the core idea of the topic.
- **Key terms** in bold, each followed by a one-line definition.

Repeat one "## [Topic name]" section per major topic found in the slides.
Only use content that appears in the source — never invent facts, numbers,
or examples that aren't there. Keep it dense: no filler phrases like
"this section discusses" or "in this part of the lecture"."""


def summarize(state: CoachState) -> dict:
    messages = [
        SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
        HumanMessage(content=f"LECTURE CONTENT:\n{state['raw_text']}"),
    ]
    response = llm.invoke(messages)
    print("[summarize] done")
    return {"summary": response.content}


NOTES_SYSTEM_PROMPT = """You are a study coach creating detailed study notes \
from lecture slides — more granular than a summary, meant for deep review \
before an exam. Follow this exact structure:

## [Topic name]
1. Numbered list of every distinct fact, definition, formula, or claim from
   the slides on this topic — one per line, self-contained (understandable
   without re-reading the slide).
2. Continue numbering through all facts for that topic.

**Watch for:** one line naming anything that looks exam-relevant (an edge
case, a formula, a term the slides emphasize) — only include this line if
something genuinely stands out.

Only use content in the source. Do not summarize or compress — this should
be MORE detailed than the summary, not less."""


def make_notes(state: CoachState) -> dict:
    messages = [
        SystemMessage(content=NOTES_SYSTEM_PROMPT),
        HumanMessage(content=f"LECTURE CONTENT:\n{state['raw_text']}"),
    ]
    response = llm.invoke(messages)
    print("[make_notes] done")
    return {"notes": response.content}
