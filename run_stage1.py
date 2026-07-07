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

    # retry_round=0 is passed explicitly to reset it for this fresh sitting.
    # weak_topics is deliberately NOT passed here — leaving it out means
    # whatever was last saved for this thread_id carries forward untouched.
    result = graph.invoke(
        {"file_path": pdf_path, "steps": steps, "retry_round": 0},
        config=config,
    )

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
        if result.get("weak_topics"):
            print(f"Still weak on: {', '.join(result['weak_topics'])} "
                  f"— saved to memory, next quiz session will retest these first.")
        else:
            print("No weak topics remaining — nice work! (Cleared from memory too.)")
