import streamlit as st
import requests

st.set_page_config(page_title="RAG Chatbot 💬", layout="centered")

st.title("📄 RAG Chatbot")
st.markdown("Upload a `.pdf` or `.txt` resume, then ask questions!")

API_URL = "http://localhost:8000/api"

# Upload Section
st.subheader("📤 Upload Document")
uploaded_files = st.file_uploader("Choose a file", type=["pdf", "txt"], accept_multiple_files=True)

if uploaded_files:
    files = [("files", (f.name, f.read(), f.type)) for f in uploaded_files]
    res = requests.post(f"{API_URL}/upload/", files=files)
    if res.status_code == 200:
        st.success(f"✅ Uploaded and chunked {res.json()['chunks']} chunks.")
    else:
        st.error("❌ Upload failed")

# Chat Section
st.subheader("💬 Ask a Question")
query = st.text_input("Enter your question")

if st.button("Ask"):
    if query:
        with st.spinner("Submitting your question and waiting for the answer..."):
            res = requests.get(f"{API_URL}/query/", params={"q": query})
            if res.status_code == 200:
                result = res.json()
                st.markdown(f"**🤖 Answer:** {result['answer']}")
                if result["mode"] == "no-context":
                    st.warning("⚠️ Answer generated without resume context (LLM fallback)")
                else:
                    st.success("📄 Answer generated using document context (RAG)")
            else:
                st.error("Failed to get a response")
    else:
        st.info("Please enter a question.")