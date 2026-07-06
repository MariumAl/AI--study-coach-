"""
Run this: python3 run_stage1.py path/to/your_lecture.pdf [summary|notes|both]

mode defaults to "both" if you don't pass one.
"""

import sys
from dotenv import load_dotenv

load_dotenv()

from graph import graph

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 run_stage1.py path/to/your_lecture.pdf [summary|notes|both]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "both"

    if mode not in ("summary", "notes", "both"):
        print(f"Invalid mode '{mode}'. Must be one of: summary, notes, both")
        sys.exit(1)

    # .invoke() runs the graph start to finish. Nodes that never ran (because
    # the router skipped them) simply won't have added their key to result —
    # so we use .get() rather than result["..."] to avoid a KeyError.
    result = graph.invoke({"file_path": pdf_path, "mode": mode})

    if result.get("summary"):
        print("\n--- SUMMARY ---\n")
        print(result["summary"])

    if result.get("notes"):
        print("\n--- DETAILED NOTES ---\n")
        print(result["notes"])
