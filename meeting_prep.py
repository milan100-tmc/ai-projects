import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import numpy as np
import tempfile
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="Meeting Prep AI", page_icon="🧠", layout="wide")

with st.sidebar:
    st.title("🧠 Meeting Prep AI")
    st.caption("Upload your report. Walk into every meeting prepared.")
    st.divider()
    uploaded_file = st.file_uploader("Upload report (PDF)", type=["pdf"])
    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

st.title("🧠 Meeting Prep AI")
st.caption("No more walking into meetings unprepared. Upload your report, get briefed in 60 seconds.")

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
    vectorizer = TfidfVectorizer(max_features=384)
    matrix = vectorizer.fit_transform(texts).toarray().astype('float32')
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return texts, vectorizer, index

def search(query, chunks, vectorizer, index, k=3):
    vec = vectorizer.transform([query]).toarray().astype('float32')
    faiss.normalize_L2(vec)
    _, ids = index.search(vec, k=k)
    return [chunks[i] for i in ids[0] if i < len(chunks)]

def ask_groq(messages):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    return response.choices[0].message.content

if "messages" not in st.session_state:
    st.session_state.messages = []
if "brief_generated" not in st.session_state:
    st.session_state.brief_generated = False

if uploaded_file:
    with st.spinner("Reading report and building index..."):
        chunks, vectorizer, index = process_pdf(uploaded_file.getvalue())
    st.success(f"✅ {uploaded_file.name} ready — {len(chunks)} sections indexed")

    # Auto-generate key questions on first upload
    if not st.session_state.brief_generated:
        with st.spinner("Generating your pre-meeting brief..."):
            full_text = " ".join(chunks[:10])
            questions = ask_groq([
                {"role": "system", "content": "You are a business intelligence assistant helping a manager prepare for a meeting."},
                {"role": "user", "content": f"Based on this business report, generate 5 sharp questions a manager should ask in the meeting:\n\n{full_text}"}
            ])
            summary = ask_groq([
                {"role": "system", "content": "You are a business intelligence assistant."},
                {"role": "user", "content": f"Summarise this business report in 3 bullet points a busy manager needs to know:\n\n{full_text}"}
            ])

        st.subheader("📋 Your Pre-Meeting Brief")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Key Takeaways**")
            st.markdown(summary)
        with col2:
            st.markdown("**Questions to Ask**")
            st.markdown(questions)
        st.divider()
        st.session_state.brief_generated = True

    st.subheader("💬 Ask anything about this report")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a follow-up question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        relevant = search(prompt, chunks, vectorizer, index)
        context = "\n".join(relevant)
        history = st.session_state.messages[-6:]
        messages = [
            {"role": "system", "content": f"You are a business intelligence assistant. Answer based on this report context:\n{context}"}
        ] + history

        with st.chat_message("assistant"):
            reply = ask_groq(messages)
            st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})

else:
    st.info("👈 Upload a business report in the sidebar to get your pre-meeting brief.")
    st.markdown("### How it works")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. Upload**\nUpload any business report, financial summary or meeting document")
    with col2:
        st.markdown("**2. Get Briefed**\nGet key takeaways and smart questions automatically generated")
    with col3:
        st.markdown("**3. Ask**\nAsk follow-up questions in plain English before walking in")
