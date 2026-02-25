import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import tempfile

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="Business Intel Assistant", page_icon="📊", layout="wide")

with st.sidebar:
    st.title("📊 Business Intel Assistant")
    st.caption("Upload a business report and ask questions in plain English.")
    st.divider()
    uploaded_file = st.file_uploader("Upload a PDF report", type=["pdf"])
    st.divider()
    st.markdown("**Example questions:**")
    st.markdown("- What are the key findings?")
    st.markdown("- Summarise the main risks")
    st.markdown("- What actions were recommended?")

st.title("Ask your business documents anything")
st.caption("No more 2-hour meetings to understand a report. Just ask.")

@st.cache_resource
def process_pdf(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    return [c.page_content for c in chunks]

if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file:
    with st.spinner("Reading document..."):
        chunks = process_pdf(uploaded_file.getvalue())
    st.success(f"✅ Ready — {len(chunks)} sections loaded from {uploaded_file.name}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        prompt_words = prompt.lower().split()
        scored = []
        for chunk in chunks:
            score = sum(1 for w in prompt_words if w in chunk.lower())
            scored.append((score, chunk))
        scored.sort(reverse=True)
        context = "\n".join([c for _, c in scored[:3]])

        with st.chat_message("assistant"):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": f"You are a business intelligence assistant. Answer clearly and concisely based on this document context:\n{context}"},
                    {"role": "user", "content": prompt}
                ]
            )
            reply = response.choices[0].message.content
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
else:
    st.info("👈 Upload a PDF in the sidebar to get started.")
