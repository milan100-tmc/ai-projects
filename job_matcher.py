import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask(prompt, system="You are an expert recruiter and career coach. Be specific, direct and actionable."):
    return client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    ).choices[0].message.content

st.set_page_config(page_title="Job Application Assistant", page_icon="🎯", layout="wide")

st.title("🎯 Job Application Assistant")
st.caption("Paste your CV and a job description. Get everything you need to apply in 60 seconds.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Your CV")
    cv_text = st.text_area("Paste your CV here", height=400, placeholder="Paste your full CV text...")
with col2:
    st.subheader("Job Description")
    jd_text = st.text_area("Paste the job description here", height=400, placeholder="Paste the full job description...")

st.divider()

if st.button("🚀 Generate Full Application Package", type="primary"):
    if not cv_text or not jd_text:
        st.error("Please paste both your CV and the job description.")
    else:
        # Run all 5 analyses
        with st.spinner("Analysing match..."):
            analysis = ask(f"""Analyse this CV against this job description.

CV: {cv_text}

Job Description: {jd_text}

Provide exactly:
1. MATCH SCORE (0-100) with one line explanation
2. TOP 3 STRENGTHS
3. TOP 3 GAPS
4. ONE SENTENCE SUMMARY of what the hiring manager will think""")

        with st.spinner("Running ATS check..."):
            ats = ask(f"""You are an ATS system. Analyse this CV against this job description.

CV: {cv_text}

Job Description: {jd_text}

Provide:
1. ATS SCORE (0-100)
2. CRITICAL KEYWORDS MISSING — list every important keyword from the JD missing in the CV
3. KEYWORDS PRESENT — list matching keywords
4. FORMAT ISSUES — any CV formatting that would hurt ATS parsing
5. QUICK FIXES — 3 specific changes to improve ATS score immediately""")

        with st.spinner("Rewriting CV summary..."):
            rewrite = ask(f"""Rewrite this candidate's CV professional summary and 5 bullet points tailored specifically for this job.

CV: {cv_text}

Job Description: {jd_text}

Provide:
1. REWRITTEN PROFESSIONAL SUMMARY (3 sentences, mirror JD language, include key metrics)
2. 5 REWRITTEN BULLET POINTS (strong action verbs, numbers, mirror JD keywords)

Be specific. Use exact language from the job description.""")

        with st.spinner("Predicting interview questions..."):
            questions = ask(f"""Based on this job description and CV, predict the 8 most likely interview questions.

CV: {cv_text}

Job Description: {jd_text}

For each question provide:
- The question
- Why they will ask it (one line)
- A strong answer framework (2-3 sentences using the candidate's actual experience)

Focus on questions that probe the gaps between the CV and JD.""")

        with st.spinner("Writing cold email..."):
            email = ask(f"""Write a cold email from this candidate to the hiring manager for this role.

CV: {cv_text}

Job Description: {jd_text}

Rules:
- Subject line that gets opened
- 4 sentences maximum
- First sentence: specific hook about the company/role
- Second sentence: single strongest qualification with a number
- Third sentence: one concrete thing they built or achieved
- Fourth sentence: clear call to action
- No generic phrases
- Sound human not corporate""", 
system="You are an expert at writing cold emails that get responses. Be punchy and specific.")

        # Display results in tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Match Analysis", 
            "🤖 ATS Score", 
            "✏️ Rewritten CV",
            "❓ Interview Prep",
            "📧 Cold Email"
        ])

        with tab1:
            st.markdown(analysis)

        with tab2:
            st.markdown(ats)

        with tab3:
            st.markdown(rewrite)
            st.info("Copy these into your CV before applying. Mirror the exact language of the job description.")

        with tab4:
            st.markdown(questions)
            st.info("Prepare answers to these before every interview.")

        with tab5:
            st.markdown(email)
            st.info("Find the hiring manager on LinkedIn and send this directly. Don't wait for the application process.")
