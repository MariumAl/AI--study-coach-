# AI Study Coach

An agentic study coach built with LangChain + LangGraph, as a learning project.
Upload lecture slides (PDF) and it generates a structured summary and/or
detailed notes. Flashcards, quizzes, and adaptive weak-topic tracking are
in progress.

## Current pipeline (Stage 1)

```
read_pdf → [decision: mode] → summarize → [decision: mode] → make_notes
```

`mode` (`summary` / `notes` / `both`) controls which nodes actually run,
via LangGraph conditional edges — no code editing required between runs.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # paste your GOOGLE_API_KEY (or swap to Anthropic, see nodes.py)
```

## Run

```bash
python run_stage1.py path/to/your_lecture.pdf summary
python run_stage1.py path/to/your_lecture.pdf notes
python run_stage1.py path/to/your_lecture.pdf both
```

## Roadmap
- [x] Stage 1: read PDF → summarize → notes, with conditional routing
- [ ] Stage 2: flashcards + quiz generation (structured output)
- [ ] Stage 3: evaluate answers, track weak topics in state
- [ ] Stage 4: persist state across sessions (checkpointing) so "quiz me"
      adapts to weak chapters from previous runs — the core agentic/memory piece

