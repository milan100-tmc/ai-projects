import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="CV Job Matcher", page_icon="🎯", layout="wide")

st.title("🎯 CV & Job Match Analyzer")
st.caption("Paste your CV and a job description. Get a match score, gap analysis and rewritten bullet points.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Your CV")
    cv_text = st.text_area("Paste your CV here", height=400, placeholder="Paste your full CV text...")

with col2:
    st.subheader("Job Description")
    jd_text = st.text_area("Paste the job description here", height=400, placeholder="Paste the full job description...")

st.divider()

if st.button("🔍 Analyse Match", type="primary"):
    if not cv_text or not jd_text:
        st.error("Please paste both your CV and the job description.")
    else:
        with st.spinner("Analysing your match..."):

            # Match score and gap analysis
            analysis = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an expert recruiter and career coach. Be specific, direct and actionable."},
                    {"role": "user", "content": f"""Analyse this CV against this job description.

CV:
{cv_text}

Job Description:
{jd_text}

Provide:
1. MATCH SCORE (0-100) with one line explanation
2. TOP 3 STRENGTHS — what in the CV matches well
3. TOP 3 GAPS — what's missing or weak
4. KEYWORDS MISSING — list the important keywords from the JD not in the CV
5. ONE SENTENCE SUMMARY — what the hiring manager will think when they see this CV
"""}
                ]
            ).choices[0].message.content

            # Rewritten bullet points
            rewrite = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an expert CV writer. Be specific and use strong action verbs."},
                    {"role": "user", "content": f"""Based on this job description, rewrite 4 bullet points for this candidate's CV that would maximize their chances.

CV:
{cv_text}

Job Description:
{jd_text}

Rules:
- Start each bullet with a strong action verb
- Include numbers/metrics where possible
- Mirror the language of the job description
- Keep each bullet under 20 words
"""}
                ]
            ).choices[0].message.content

            # Cover letter opening
            cover = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are an expert cover letter writer."},
                    {"role": "user", "content": f"""Write a powerful 3-sentence cover letter opening for this candidate applying to this role.

CV:
{cv_text}

Job Description:
{jd_text}

Make it specific, confident and human. No generic phrases like 'I am writing to apply'.
"""}
                ]
            ).choices[0].message.content

        # Display results
        st.subheader("📊 Match Analysis")
        st.markdown(analysis)

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("✏️ Rewritten CV Bullet Points")
            st.markdown(rewrite)

        with col2:
            st.subheader("📝 Cover Letter Opening")
            st.markdown(cover)
            st.caption("Use this as the opening paragraph of your cover letter.")
