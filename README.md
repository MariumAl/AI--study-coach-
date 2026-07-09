# AI Study Coach

An agentic study coach built with LangChain + LangGraph, as a learning project.
Upload lecture slides (PDF) and it generates a summary, detailed notes,
flashcards, and a quiz — then grades your answers, tracks which topics
you're weak on, and adaptively re-quizzes you on just those topics. Weak
topics persist across separate runs (via a LangGraph checkpointer), so
running "quiz me" again tomorrow remembers what you struggled with today.

## Pipeline

```
read_pdf → [optional: summarize] → [optional: make_notes]
         → [optional: flashcards] → [optional: quiz]

quiz → ask_answers → evaluate → (loop back to quiz if weak topics remain,
                                  up to 2 retry rounds) → end
```

Which steps run is controlled entirely at runtime (no code edits) via a
comma-separated `steps` argument: `summary`, `notes`, `flashcards`, `quiz`,
in any combination. LangGraph conditional edges decide the routing.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Currently configured to run on a **local Ollama model** (`qwen2.5:7b`) —
no API key needed, just make sure Ollama is running. To swap to Claude or
Gemini instead, see the top of `nodes.py` (`.env.example` has both key
formats ready).

## Run

**CLI:**
```bash
python run_stage1.py path/to/your_lecture.pdf summary
python run_stage1.py path/to/your_lecture.pdf notes,flashcards
python run_stage1.py path/to/your_lecture.pdf quiz
python run_stage1.py path/to/your_lecture.pdf summary,notes,flashcards,quiz
```

**Web UI (Streamlit):**
```bash
streamlit run app.py
```
Upload a PDF, pick which steps to generate, hit Run. If quiz is included,
answer questions in the form that appears — same underlying graph as the
CLI, just a different caller for the interrupt/resume cycle.

Each PDF gets its own memory "thread" (by filename), in both the CLI and
the web UI — run the same PDF again later with quiz included, and it
picks up right where your weak topics left off.

## Roadmap
- [x] Stage 1: read PDF → summarize / notes, conditional routing
- [x] Stage 2: flashcards + quiz generation (structured output via Pydantic)
- [x] Stage 3: evaluate answers (LLM-as-judge), track weak topics, adaptive
      retry loop (the core agentic control-flow piece)
- [x] Stage 4: persist state across sessions (SQLite checkpointer) — "quiz
      me" remembers weak chapters from previous runs
- [x] Stage 5: Streamlit frontend (uses `interrupt()`/`Command(resume=...)`
      instead of `input()`, so the same graph works from a browser too)
- [x] Stage 6: `make_notes` can autonomously call a web_search tool when the
      slides mention a concept too briefly to write a clear note about it —
      real tool-use agency inside a single node (same manual tool-loop
      pattern as the very first stage1_plain_langchain.py experiment)
- [ ] Next: more features (TBD)
