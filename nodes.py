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
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from state import CoachState

# No API key needed — this talks to Ollama's local server (default
# http://localhost:11434). Make sure Ollama is running before you invoke
# the graph, or you'll get a connection error, not an auth error.
llm = ChatOllama(model="qwen2.5:7b", temperature=0)


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


quiz_llm = llm.with_structured_output(QuizSet)

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
