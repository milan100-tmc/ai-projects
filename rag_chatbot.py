import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.title("📄 Milan's Business Intelligence Assistant")
st.write("Ask me anything from your documents!")

@st.cache_resource
def load_documents():
    docs_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
    all_text = []
    for filename in os.listdir(docs_folder):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(docs_folder, filename))
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(pages)
            all_text.extend([c.page_content for c in chunks])
    return all_text

with st.spinner("Loading your documents..."):
    chunks = load_documents()
    st.success(f"✅ {len(chunks)} chunks loaded from your documents!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Simple keyword search instead of embeddings
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
                {"role": "system", "content": f"You are a helpful business intelligence assistant. Answer questions based on this document context:\n{context}"},
                {"role": "user", "content": prompt}
            ]
        )
        reply = response.choices[0].message.content
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
