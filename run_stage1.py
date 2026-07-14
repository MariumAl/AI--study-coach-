"""
Run this: python3 run_stage1.py path/to/your_lecture.pdf [steps]

[steps] is a comma-separated list from: summary, notes, flashcards, quiz
Any combination works, e.g.:
    python3 run_stage1.py lecture.pdf summary
    python3 run_stage1.py lecture.pdf notes,flashcards
    python3 run_stage1.py lecture.pdf quiz

If "quiz" is included, you'll be asked to answer each question in the
terminal. The graph grades you, and if any topics are weak, it loops back
with new questions focused on just those topics (up to 2 retry rounds).

MEMORY ACROSS RUNS: each PDF gets its own "thread_id" (derived from the
filename). Run this script again later on the SAME pdf with steps=quiz,
and weak_topics from your last session carries over automatically —
that's the checkpointer at work, not anything this script does manually.
retry_round IS reset to 0 on every fresh run, though: the retry cap is
meant to limit one sitting, not accumulate across days.
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()

from graph import graph
from langgraph.types import Command

VALID_STEPS = {"summary", "notes", "flashcards", "quiz"}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 run_stage1.py path/to/your_lecture.pdf [steps]")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if len(sys.argv) > 2:
        steps = [s.strip() for s in sys.argv[2].split(",") if s.strip()]
    else:
        steps = ["summary", "notes", "flashcards", "quiz"]

    invalid = set(steps) - VALID_STEPS
    if invalid:
        print(f"Invalid step(s): {invalid}. Must be from: {VALID_STEPS}")
        sys.exit(1)
    if not steps:
        print("No steps requested — nothing to do.")
        sys.exit(1)

    # One memory "thread" per PDF filename — a different lecture file gets
    # its own independent history of weak topics, past quiz results, etc.
    thread_id = os.path.splitext(os.path.basename(pdf_path))[0]
    config = {"configurable": {"thread_id": thread_id}}

    # retry_round=0 and notes_reviewed=False reset both per-session limits
    # for a fresh sitting. weak_topics is deliberately NOT passed here —
    # leaving it out means whatever was last saved for this thread_id
    # carries forward untouched.
    result = graph.invoke(
        {"file_path": pdf_path, "steps": steps, "retry_round": 0, "notes_reviewed": False},
        config=config,
    )

    # If the graph paused at ask_answers, result contains "__interrupt__"
    # instead of the normal state. Keep resuming until it finishes for
    # real — this can loop more than once, since the supervisor can send
    # the student through review_notes before another quiz round.
    last_shown_review = None
    while "__interrupt__" in result:
        # The supervisor may have routed through review_notes before this
        # round — show that focused re-explanation before the new quiz.
        if result.get("focused_review") and result["focused_review"] != last_shown_review:
            print("\n--- FOCUSED REVIEW (before retesting) ---\n")
            print(result["focused_review"])
            last_shown_review = result["focused_review"]

        quiz_payload = result["__interrupt__"][0].value["quiz"]
        print("\n--- QUIZ TIME ---")
        answers = []
        for i, q in enumerate(quiz_payload, start=1):
            print(f"\n{i}. [{q['topic']}] {q['question']}")
            answers.append(input("Your answer: "))
        result = graph.invoke(Command(resume=answers), config=config)

    if result.get("summary"):
        print("\n--- SUMMARY ---\n")
        print(result["summary"])

    if result.get("notes"):
        print("\n--- DETAILED NOTES ---\n")
        print(result["notes"])

    if result.get("flashcards"):
        print(f"\n--- FLASHCARDS ({len(result['flashcards'])}) ---\n")
        for i, card in enumerate(result["flashcards"], start=1):
            print(f"{i}. [{card['topic']}] Q: {card['question']}")
            print(f"   A: {card['answer']}\n")

    if "quiz" in steps:
        print("\n--- QUIZ RESULTS ---")
        print(f"Practice rounds this session: {result.get('retry_round', 0)}")
        if result.get("supervisor_reasoning"):
            print(f"Supervisor's call: {result['supervisor_reasoning']}")
        if result.get("weak_topics"):
            print(f"Still weak on: {', '.join(result['weak_topics'])} "
                  f"— saved to memory, next quiz session will retest these first.")
        else:
            print("No weak topics remaining — nice work! (Cleared from memory too.)")
