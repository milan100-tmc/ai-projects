import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import tempfile
import faiss
import numpy as np
from openai import OpenAI

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy"))

def get_embeddings(texts):
    """Use Groq-compatible embeddings via a simple TF-IDF approach"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=384)
    matrix = vectorizer.fit_transform(texts)
    return matrix.toarray(), vectorizer

def search_chunks(query, chunks, vectorizer, index):
    query_vec = vectorizer.transform([query]).toarray().astype('float32')
    faiss.normalize_L2(query_vec)
    _, indices = index.search(query_vec, k=3)
    return [chunks[i] for i in indices[0] if i < len(chunks)]

st.set_page_config(page_title="Business Intel Assistant", page_icon="📊", layout="wide")

with st.sidebar:
    st.title("📊 Business Intel Assistant")
    st.caption("Upload a business report and ask questions in plain English.")
    st.divider()
    uploaded_file = st.file_uploader("Upload a PDF report", type=["pdf"])
    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()
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
    texts = [c.page_content for c in chunks]
    matrix, vectorizer = get_embeddings(texts)
    matrix = matrix.astype('float32')
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return texts, vectorizer, index

if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file:
    with st.spinner("Building vector index..."):
        chunks, vectorizer, index = process_pdf(uploaded_file.getvalue())
    st.success(f"✅ Vector index ready — {len(chunks)} sections from {uploaded_file.name}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        relevant = search_chunks(prompt, chunks, vectorizer, index)
        context = "\n".join(relevant)

        history = st.session_state.messages[-6:]
        messages = [
            {"role": "system", "content": f"You are a business intelligence assistant. Answer clearly based on this document context:\n{context}"}
        ] + history

        with st.chat_message("assistant"):
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages
            )
            reply = response.choices[0].message.content
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})
else:
    st.info("👈 Upload a PDF in the sidebar to get started.")
