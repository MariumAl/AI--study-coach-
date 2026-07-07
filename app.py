"""
Streamlit frontend for the study coach.

Run with: streamlit run app.py

KEY CONCEPT — st.session_state:
Streamlit reruns this ENTIRE file top-to-bottom every time you click
anything (a button, a form submit). A normal Python variable would just
reset to nothing on the next rerun. Anything that needs to survive between
interactions — the graph's config (thread_id), the latest result, which
quiz round we're mid-way through — has to live in st.session_state instead,
which Streamlit preserves across reruns for you.

Notice this file imports the exact same `graph` as run_stage1.py, and the
exact same interrupt()/Command(resume=...) pattern. The CLI and this app
are just two different callers of the same underlying graph — nothing in
graph.py or nodes.py needed to change to support a second frontend.
"""

import os
import streamlit as st
from graph import graph
from langgraph.types import Command

st.set_page_config(page_title="AI Study Coach", page_icon="📚")
st.title("📚 AI Study Coach")

UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- session_state setup (persists across reruns) ------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "config" not in st.session_state:
    st.session_state.config = None

# --- inputs ---------------------------------------------------------------
uploaded_file = st.file_uploader("Upload your lecture slides (PDF)", type="pdf")

steps = st.multiselect(
    "What do you want generated?",
    options=["summary", "notes", "flashcards", "quiz"],
    default=["summary", "notes", "flashcards", "quiz"],
)

if uploaded_file and st.button("Run", type="primary"):
    # Save to disk with a stable filename so the thread_id (and therefore
    # the memory tied to it) matches across separate sessions, same as the
    # CLI script keying memory off the PDF's filename.
    pdf_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    thread_id = os.path.splitext(uploaded_file.name)[0]
    config = {"configurable": {"thread_id": thread_id}}
    st.session_state.config = config

    with st.spinner("Thinking..."):
        result = graph.invoke(
            {"file_path": pdf_path, "steps": steps, "retry_round": 0},
            config=config,
        )
    st.session_state.result = result

result = st.session_state.result

# --- case 1: graph is paused waiting for quiz answers ---------------------
if result and "__interrupt__" in result:
    quiz_payload = result["__interrupt__"][0].value["quiz"]
    st.subheader("📝 Quiz Time")

    with st.form("quiz_form"):
        answers = []
        for i, q in enumerate(quiz_payload, start=1):
            st.markdown(f"**{i}. [{q['topic']}]** {q['question']}")
            answers.append(st.text_input("Your answer", key=f"answer_{i}"))
        submitted = st.form_submit_button("Submit Answers")

    if submitted:
        with st.spinner("Grading..."):
            # Resume the graph exactly where it paused, feeding in the
            # answers just collected from the form.
            new_result = graph.invoke(Command(resume=answers), config=st.session_state.config)
        st.session_state.result = new_result
        st.rerun()  # rerun the script now that state has moved forward

# --- case 2: graph finished (fully, or this round of quiz) ----------------
elif result:
    if result.get("summary"):
        with st.expander("📄 Summary", expanded=True):
            st.markdown(result["summary"])

    if result.get("notes"):
        with st.expander("📝 Detailed Notes"):
            st.markdown(result["notes"])

    if result.get("flashcards"):
        with st.expander(f"🗂️ Flashcards ({len(result['flashcards'])})"):
            for i, card in enumerate(result["flashcards"], start=1):
                st.markdown(f"**{i}. [{card['topic']}]** {card['question']}")
                st.markdown(f"> {card['answer']}")

    if "quiz" in steps and "retry_round" in result:
        st.subheader("🎯 Quiz Results")
        st.write(f"Practice rounds this session: {result.get('retry_round', 0)}")
        if result.get("weak_topics"):
            st.warning(
                f"Still weak on: {', '.join(result['weak_topics'])} — "
                f"saved to memory, next quiz session will retest these first."
            )
        else:
            st.success("No weak topics remaining — nice work! (Cleared from memory too.)")
