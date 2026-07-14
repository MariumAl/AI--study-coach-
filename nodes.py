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
from typing import List
from pydantic import BaseModel, Field
from pypdf import PdfReader
from duckduckgo_search import DDGS
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from state import CoachState

# No API key needed — this talks to Ollama's local server (default
# http://localhost:11434). Make sure Ollama is running before you invoke
# the graph, or you'll get a connection error, not an auth error.
llm = ChatOllama(model="qwen2.5:7b", temperature=0)

# A SECOND instance, same model, but warmer. This is the Quiz Writer
# agent's model: higher temperature means more varied, less repetitive
# question phrasing each time it's asked. The Grader and Supervisor stay
# on the cold `llm` above — consistency matters more than variety for
# judgment calls, but variety helps for content generation. This is the
# kind of per-role tuning a single shared model can't give you.
creative_llm = ChatOllama(model="qwen2.5:7b", temperature=0.7)


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

The lecture slides are your PRIMARY source — base your notes on them, not
on outside knowledge. However, if the slides mention a term, formula, or
concept too briefly to write a clear, self-contained note about it (e.g.
they name something without defining it, or use a term you're not fully
sure about), you may call web_search to verify or clarify it before
writing that note. Don't search for things the slides already explain
clearly — only use it to fill a genuine gap. Do not summarize or compress
— this should be MORE detailed than the summary, not less."""


@tool
def web_search(query: str) -> str:
    """Search the web for a concept, term, or formula that the lecture
    slides mention but don't fully explain. Returns a few short result
    snippets. Use this ONLY to clarify something the source material left
    unclear — not as a replacement for the lecture content itself."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
    if not results:
        return "No results found."
    return "\n\n".join(f"Title: {r['title']}\nSnippet: {r['body']}" for r in results)


# A SEPARATE bound model for this node — bind_tools() only tells the model
# what it's ALLOWED to call, it doesn't affect the plain `llm` used
# elsewhere (summarize, flashcards, etc). Each node can give the model a
# different set of capabilities.
notes_llm = llm.bind_tools([web_search])


def make_notes(state: CoachState) -> dict:
    """Same manual tool-calling loop as stage1_plain_langchain.py: invoke
    the model, check if it asked for a tool, run the tool ourselves if so,
    feed the result back, repeat until it's ready to give a final answer.
    The difference from Stage 1 is that this loop now lives inside ONE
    node of a bigger graph, instead of being the entire program — this is
    exactly how real agentic systems compose: small tool-using loops
    nested inside a larger orchestrated pipeline.
    """
    messages = [
        SystemMessage(content=NOTES_SYSTEM_PROMPT),
        HumanMessage(content=f"LECTURE CONTENT:\n{state['raw_text']}"),
    ]

    while True:
        response = notes_llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            print("[make_notes] done")
            return {"notes": response.content}

        for call in response.tool_calls:
            print(f"[make_notes] model chose to search: {call['args'].get('query')}")
            result = web_search.invoke(call["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))


# --- Structured output: flashcards -------------------------------------
# Instead of asking for free text and hoping it looks like flashcards, we
# describe the EXACT shape we want as a Pydantic model. LangChain then
# constrains the model's output to match this schema and parses it back
# into real Python objects (Flashcard instances), not a string.

class Flashcard(BaseModel):
    topic: str = Field(description="Short topic/chapter this card belongs to")
    question: str = Field(description="The question side of the flashcard")
    answer: str = Field(description="The answer side of the flashcard")


class FlashcardSet(BaseModel):
    flashcards: List[Flashcard]


# .with_structured_output() returns a NEW runnable that behaves like the
# model but guarantees its output matches the schema above.
flashcard_llm = llm.with_structured_output(FlashcardSet)

FLASHCARD_SYSTEM_PROMPT = """You are a study coach creating flashcards from \
lecture slides. Create one flashcard per distinct fact, definition, or \
concept — enough to cover everything a student needs to memorize for an \
exam. Questions should be specific and testable, not vague ("What is X?" \
is fine; "Tell me about the lecture" is not). Only use content from the \
source material."""


def make_flashcards(state: CoachState) -> dict:
    messages = [
        SystemMessage(content=FLASHCARD_SYSTEM_PROMPT),
        HumanMessage(content=f"LECTURE CONTENT:\n{state['raw_text']}"),
    ]
    result: FlashcardSet = flashcard_llm.invoke(messages)
    print(f"[make_flashcards] generated {len(result.flashcards)} flashcards")
    # Convert Pydantic objects to plain dicts to store in state (state is a
    # TypedDict of plain Python types — keep it simple/serializable).
    return {"flashcards": [card.model_dump() for card in result.flashcards]}


# --- Structured output: quiz ---------------------------------------------
# Same pattern as flashcards, but a quiz question needs a correct_answer
# field — this is what the (not-yet-built) evaluator will compare the
# student's typed answer against.

class QuizQuestion(BaseModel):
    topic: str = Field(description="Short topic/chapter this question tests")
    question: str = Field(description="The quiz question text")
    correct_answer: str = Field(
        description="A concise correct answer, used to grade the student's response"
    )


class QuizSet(BaseModel):
    questions: List[QuizQuestion]


quiz_llm = creative_llm.with_structured_output(QuizSet)

QUIZ_SYSTEM_PROMPT = """You are a study coach creating quiz questions from \
lecture slides. Each question needs a concise, gradable correct_answer (a \
sentence or two, not an essay). Only use content from the source material."""


def make_quiz(state: CoachState) -> dict:
    weak_topics = state.get("weak_topics") or []

    if weak_topics:
        # Retry round: focus narrowly on what the student got wrong, instead
        # of a fresh quiz on everything. This is the "adaptive" part — the
        # prompt itself changes based on what the graph has learned so far.
        topic_list = ", ".join(weak_topics)
        instruction = (
            f"The student struggled with these specific topics: {topic_list}. "
            f"Create 3-4 NEW questions (not repeats) that specifically retest "
            f"understanding of these topics. Do not cover anything else."
        )
        print(f"[make_quiz] adaptive round — focusing on: {topic_list}")
    else:
        instruction = (
            "Create 5-8 questions covering the most important concepts — mix "
            "definition questions (\"What is X?\") with applied ones (\"Why "
            "does X happen?\" or \"What would happen if...?\")."
        )
        print("[make_quiz] initial quiz — covering all topics")

    messages = [
        SystemMessage(content=QUIZ_SYSTEM_PROMPT),
        HumanMessage(
            content=f"{instruction}\n\nLECTURE CONTENT:\n{state['raw_text']}"
        ),
    ]
    result: QuizSet = quiz_llm.invoke(messages)
    print(f"[make_quiz] generated {len(result.questions)} quiz questions")
    return {"quiz": [q.model_dump() for q in result.questions]}


from langgraph.types import interrupt


def ask_answers(state: CoachState) -> dict:
    """Pause the graph here and hand the quiz questions back to whatever
    is calling it (CLI script or Streamlit app). Execution stops at this
    exact point until the caller invokes the graph again with
    Command(resume=<the student's answers>) — at which point `interrupt()`
    below returns that value and the node continues.

    This replaces the old input()-based version: input() only works in a
    terminal, but interrupt() works the same way whether the caller is a
    CLI script or a web UI — the node itself doesn't need to know which.
    """
    answers = interrupt({"quiz": state["quiz"]})
    return {"answers": answers}


class Grade(BaseModel):
    correct: bool = Field(description="Whether the student's answer is substantively correct")
    feedback: str = Field(description="One sentence of feedback, especially if wrong")


grade_llm = llm.with_structured_output(Grade)

GRADE_SYSTEM_PROMPT = """You are grading a student's quiz answer. Judge \
based on MEANING, not exact wording — "hash map" and "hash table" should \
both count as correct if the question is about that concept. Be reasonably \
lenient: partial understanding of the core idea counts as correct. Only \
mark it wrong if the core concept is actually missing or incorrect."""


def evaluate(state: CoachState) -> dict:
    """Grade every answer, and collect which TOPICS the student got wrong.
    This is the state that turns into 'memory' for the adaptive loop —
    make_quiz reads weak_topics on the next round to decide what to retest.

    Feedback is built as a plain list of dicts and RETURNED as state, not
    just printed. print() only reaches whatever terminal launched this
    Python process — a Streamlit app's browser never sees it. Anything a
    frontend needs to display has to travel through state, same as
    everything else in this graph.
    """
    weak_topics = []
    feedback_list = []
    correct_count = 0

    for q, student_answer in zip(state["quiz"], state["answers"]):
        # Don't trust the LLM to correctly handle a trivial edge case like an
        # empty answer — decide that deterministically in plain Python first.
        if not student_answer.strip():
            weak_topics.append(q["topic"])
            feedback_list.append({
                "topic": q["topic"], "question": q["question"],
                "student_answer": student_answer, "correct": False,
                "feedback": "No answer provided.",
            })
            print(f"  ✗ [{q['topic']}] No answer provided.")
            continue

        messages = [
            SystemMessage(content=GRADE_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Question: {q['question']}\n"
                f"Correct answer: {q['correct_answer']}\n"
                f"Student's answer: {student_answer}"
            )),
        ]
        grade: Grade = grade_llm.invoke(messages)

        feedback_list.append({
            "topic": q["topic"], "question": q["question"],
            "student_answer": student_answer, "correct": grade.correct,
            "feedback": grade.feedback,
        })

        if grade.correct:
            correct_count += 1
            print(f"  ✓ [{q['topic']}] correct")
        else:
            weak_topics.append(q["topic"])
            print(f"  ✗ [{q['topic']}] {grade.feedback}")

    print(f"\n[evaluate] scored {correct_count}/{len(state['quiz'])}")

    return {
        "weak_topics": weak_topics,
        "retry_round": state.get("retry_round", 0) + 1,
        "last_feedback": feedback_list,
    }


# --- MULTI-AGENT: the Supervisor -----------------------------------------
# Everything above is one specialist agent per node. The Supervisor's job
# is different in kind: it doesn't produce study content, it decides which
# specialist should act NEXT, based on the situation. This is what makes
# the system multi-agent rather than just multi-prompt — a coordinator
# making a genuine judgment call, not a fixed sequence.
#
# MAX_RETRY_ROUNDS is enforced here in plain Python, BEFORE the supervisor
# is even asked. This is a deliberate design choice: the hard safety limit
# should never depend on the model choosing to respect it — it's checked
# deterministically, and the supervisor is only consulted for the parts
# where a genuine judgment call is appropriate.
MAX_RETRY_ROUNDS = 2


class SupervisorDecision(BaseModel):
    action: str = Field(
        description='Must be exactly one of: "retest", "review_notes"'
    )
    reasoning: str = Field(description="One sentence explaining the choice")


supervisor_llm = llm.with_structured_output(SupervisorDecision)

SUPERVISOR_SYSTEM_PROMPT = """You are a tutoring supervisor. You coordinate \
two specialist agents and must choose which one acts next:

- "retest": send the student straight back to the Quiz Writer for more
  practice questions on their weak topics.
- "review_notes": send the student to the Notes Reviewer FIRST, for a
  focused re-explanation of their weak topics, before testing again.

Choose "review_notes" if the student has been wrong on the same topic(s)
before (this isn't their first miss) AND notes haven't already been
reviewed this session — a fresh explanation is more useful than blindly
re-testing something they may not understand yet. Otherwise choose
"retest". Never choose "review_notes" twice in one session."""


def supervisor(state: CoachState) -> dict:
    weak_topics = state.get("weak_topics") or []
    retry_round = state.get("retry_round", 0)

    # Deterministic hard stop — the LLM never gets a vote on this part.
    if not weak_topics or retry_round >= MAX_RETRY_ROUNDS:
        reason = (
            "No weak topics remain." if not weak_topics
            else f"Hit the {MAX_RETRY_ROUNDS}-round practice cap."
        )
        print(f"[supervisor] done — {reason}")
        return {"next_action": "done", "supervisor_reasoning": reason}

    # Only consult the LLM when there's a genuine strategic choice to make.
    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Weak topics: {', '.join(weak_topics)}\n"
            f"Retry round so far: {retry_round}\n"
            f"Notes already reviewed this session: {state.get('notes_reviewed', False)}"
        )),
    ]
    decision: SupervisorDecision = supervisor_llm.invoke(messages)
    action = decision.action if decision.action in ("retest", "review_notes") else "retest"

    # Deterministic guardrail on top of the LLM's choice: never allow
    # review_notes twice in one session, even if the model picks it anyway.
    if action == "review_notes" and state.get("notes_reviewed"):
        print("[supervisor] model chose review_notes again — overriding to retest")
        action = "retest"

    print(f"[supervisor] decided: {action} — {decision.reasoning}")
    return {"next_action": action, "supervisor_reasoning": decision.reasoning}


# --- MULTI-AGENT: the Notes Reviewer specialist --------------------------
# A second specialist, only invoked when the Supervisor delegates to it.
# Same model family as make_notes, but narrowly focused and using the
# cold `llm` (a review should be precise, not creative).

REVIEW_SYSTEM_PROMPT = """You are a study coach giving a focused, clear
re-explanation of specific topics a student just got wrong on a quiz.
Do NOT cover anything else — only the listed weak topics. For each topic:
explain the core idea in plain language (2-4 sentences), as if explaining
it for the first time to someone who was confused, not just restating the
original slide content. Base it on the lecture content provided."""


def review_notes(state: CoachState) -> dict:
    topic_list = ", ".join(state.get("weak_topics") or [])
    messages = [
        SystemMessage(content=REVIEW_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Weak topics to re-explain: {topic_list}\n\n"
            f"LECTURE CONTENT:\n{state['raw_text']}"
        )),
    ]
    response = llm.invoke(messages)
    print(f"[review_notes] wrote a focused review for: {topic_list}")
    return {"focused_review": response.content, "notes_reviewed": True}
