import streamlit as st
from rag import process_data, generate_answer

st.set_page_config(page_title="Document Q&A Bot", layout="wide")

st.title("RAG Document Q&A Assistant")

placeholder = st.empty()

# -------- Process Button --------
if st.sidebar.button("Process Documents"):
    for status in process_data():
        placeholder.text(status)

# -------- Query --------
query = st.text_input("Ask a question:")

if query:
    try:
        answer, sources = generate_answer(query)

        st.header("Answer")
        st.write(answer)

        if sources:
            st.subheader("Sources")
            for src in sources.split("\n"):
                st.markdown(f"**{src}**")

    except RuntimeError:
        st.warning("Please process documents first")

