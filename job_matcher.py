import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os
import json
import datetime

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask(prompt, system="You are an expert recruiter and career coach. Be specific, direct and actionable."):
    return client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    ).choices[0].message.content

# Load/save tracker
TRACKER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/applications.json")

def load_applications():
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, 'r') as f:
            return json.load(f)
    return []

def save_application(app):
    apps = load_applications()
    apps.append(app)
    os.makedirs(os.path.dirname(TRACKER_FILE), exist_ok=True)
    with open(TRACKER_FILE, 'w') as f:
        json.dump(apps, f, indent=2)

def update_application(index, updates):
    apps = load_applications()
    apps[index].update(updates)
    with open(TRACKER_FILE, 'w') as f:
        json.dump(apps, f, indent=2)

st.set_page_config(page_title="Job Application Assistant", page_icon="🎯", layout="wide")

# Tabs
tab_main, tab_tracker = st.tabs(["🎯 Application Assistant", "📋 Application Tracker"])

with tab_main:
    st.title("🎯 Job Application Assistant")
    st.caption("Paste your CV and a job description. Get everything you need to apply in 60 seconds.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Your CV")
        cv_text = st.text_area("Paste your CV here", height=400, placeholder="Paste your full CV text...")
    with col2:
        st.subheader("Job Description")
        jd_text = st.text_area("Paste the job description here", height=400, placeholder="Paste the full job description...")

    # Company info for tracking
    col3, col4, col5 = st.columns(3)
    with col3:
        company_name = st.text_input("Company Name", placeholder="e.g. Product Madness")
    with col4:
        job_title = st.text_input("Job Title", placeholder="e.g. Data Analyst")
    with col5:
        job_url = st.text_input("Job URL", placeholder="Paste the job link")

    st.divider()

    if st.button("🚀 Generate Full Application Package", type="primary"):
        if not cv_text or not jd_text:
            st.error("Please paste both your CV and the job description.")
        else:
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
2. CRITICAL KEYWORDS MISSING
3. KEYWORDS PRESENT
4. FORMAT ISSUES
5. QUICK FIXES — 3 specific changes to improve ATS score immediately""")

            with st.spinner("Rewriting CV..."):
                rewrite = ask(f"""Rewrite this candidate's CV professional summary and 5 bullet points tailored for this job.

CV: {cv_text}

Job Description: {jd_text}

Provide:
1. REWRITTEN PROFESSIONAL SUMMARY (3 sentences, mirror JD language, include metrics)
2. 5 REWRITTEN BULLET POINTS (strong action verbs, numbers, mirror JD keywords)""")

            with st.spinner("Predicting interview questions..."):
                questions = ask(f"""Based on this job description and CV, predict the 8 most likely interview questions.

CV: {cv_text}

Job Description: {jd_text}

For each question provide:
- The question
- Why they will ask it
- Strong answer framework using the candidate's actual experience""")

            with st.spinner("Writing cold email..."):
                email = ask(f"""Write a cold email from this candidate to the hiring manager.

CV: {cv_text}

Job Description: {jd_text}

Rules:
- Subject line that gets opened
- 4 sentences maximum
- Specific hook about the company
- Strongest qualification with a number
- One concrete achievement
- Clear call to action
- No generic phrases""",
system="You are an expert at writing cold emails that get responses. Be punchy and specific.")

            with st.spinner("Writing cover letter..."):
                cover_letter = ask(f"""Write a outstanding cover letter for this candidate for this specific role.

CV: {cv_text}

Job Description: {jd_text}

Candidate background to weave in:
- MSc Business Intelligence from Clermont School of Business
- Data Analyst Intern at OQEMA Group (chemicals supply chain) — improved reporting efficiency 30%, increased dashboard adoption 30%, reduced support queries 25%
- Data Researcher at Happy City Hub — 182 indicators, 50+ sources, 100% audit trail
- Microsoft Fabric Analytics Engineer certified
- Recently built and deployed 4 live AI tools: RAG chatbot, Meeting Prep AI, Sales Intelligence Dashboard, Supply Chain Analytics Dashboard
- Understands business problems first, builds technical solutions second
- Strong stakeholder communication — ran tutorials for non-technical users at OQEMA

Cover letter structure:
1. HEADER: Candidate full name, email, phone, LinkedIn, GitHub, date — formatted professionally
2. SALUTATION: Use hiring manager name if mentioned in JD, otherwise "Dear Hiring Manager" — never generic
3. OPENING PARAGRAPH: One powerful sentence about why THIS company specifically. Show you researched them. Then immediately state you are applying for the role.
4. SECOND PARAGRAPH: Business understanding first. Explain you understand the business problem this role solves. Reference OQEMA experience and what you learned about stakeholder adoption and business intelligence in real companies.
5. THIRD PARAGRAPH: Technical proof. Mention your recent AI projects and data engineering work as evidence you go beyond standard analyst skills. Include specific numbers from your experience.
6. FOURTH PARAGRAPH: Why you are the top candidate. Connect your unique combination of business understanding + technical skills + real deployment experience to their specific needs.
7. CLOSING: Professional closing with clear call to action. Sign off with full name.

Rules:
- Address the company by name multiple times
- Reference specific things from the job description
- Never use phrases like I am writing to apply or I believe I would be a great fit
- Sound confident not desperate
- Maximum 400 words
- Professional British English""",
system="You are an elite cover letter writer who has helped candidates get hired at top companies. Write cover letters that make hiring managers stop and call immediately.")

            # Store results in session state
            st.session_state.results = {
                'analysis': analysis,
                'ats': ats,
                'rewrite': rewrite,
                'questions': questions,
                'email': email,
                'cover_letter': cover_letter,
                'company': company_name,
                'title': job_title,
                'url': job_url,
                'cv': cv_text,
                'jd': jd_text
            }

    if 'results' in st.session_state:
        r = st.session_state.results

        t1, t2, t3, t4, t5, t6 = st.tabs([
            "📊 Match Analysis",
            "🤖 ATS Score",
            "✏️ Rewritten CV",
            "❓ Interview Prep",
            "📧 Cold Email",
            "📝 Cover Letter"
        ])

        with t1:
            st.markdown(r['analysis'])
        with t2:
            st.markdown(r['ats'])
        with t3:
            st.markdown(r['rewrite'])
            st.info("Copy these into your CV before applying.")
        with t4:
            st.markdown(r['questions'])
        with t5:
            st.markdown(r['email'])
        with t6:
            st.markdown(r['cover_letter'])

        st.divider()
        st.subheader("💾 Save to Tracker")
        col1, col2 = st.columns(2)
        with col1:
            status = st.selectbox("Status", ["Applied", "To Apply", "Interview", "Rejected", "Offer"])
        with col2:
            notes = st.text_input("Notes", placeholder="e.g. Applied via LinkedIn, emailed HR")

        if st.button("✅ Save Application", type="primary"):
            # Extract match score
            try:
                score_line = [l for l in r['analysis'].split('\n') if 'MATCH SCORE' in l or 'match score' in l.lower()]
                score = score_line[0] if score_line else "N/A"
            except:
                score = "N/A"

            app = {
                'date': datetime.datetime.now().strftime('%Y-%m-%d'),
                'company': r['company'] or 'Unknown',
                'title': r['title'] or 'Unknown',
                'url': r['url'] or '',
                'status': status,
                'match_score': score,
                'notes': notes,
                'cold_email': r['email'],
                'cover_letter': r['cover_letter'],
                'rewritten_cv': r['rewrite']
            }
            save_application(app)
            st.success(f"✅ Saved! {r['company']} — {r['title']} added to your tracker.")

with tab_tracker:
    st.title("📋 Application Tracker")
    st.caption("All your applications in one place.")

    apps = load_applications()

    if not apps:
        st.info("No applications saved yet. Generate your first application package and save it.")
    else:
        # Summary metrics
        total = len(apps)
        interviews = len([a for a in apps if a['status'] == 'Interview'])
        offers = len([a for a in apps if a['status'] == 'Offer'])
        response_rate = round((interviews + offers) / total * 100, 1) if total > 0 else 0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Applications", total)
        k2.metric("Interviews", interviews)
        k3.metric("Offers", offers)
        k4.metric("Response Rate", f"{response_rate}%")

        st.divider()

        # Filter by status
        status_filter = st.selectbox("Filter by status", ["All", "Applied", "To Apply", "Interview", "Rejected", "Offer"])

        filtered_apps = apps if status_filter == "All" else [a for a in apps if a['status'] == status_filter]

        for i, app in enumerate(reversed(filtered_apps)):
            idx = len(apps) - 1 - i
            with st.expander(f"**{app['company']}** — {app['title']} | {app['status']} | {app['date']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Company:** {app['company']}")
                    st.markdown(f"**Role:** {app['title']}")
                    st.markdown(f"**Date:** {app['date']}")
                with col2:
                    st.markdown(f"**Status:** {app['status']}")
                    st.markdown(f"**Match Score:** {app.get('match_score', 'N/A')}")
                    if app.get('url'):
                        st.markdown(f"**Job URL:** [Link]({app['url']})")
                with col3:
                    st.markdown(f"**Notes:** {app.get('notes', '-')}")

                # Update status
                new_status = st.selectbox(
                    "Update status",
                    ["Applied", "To Apply", "Interview", "Rejected", "Offer"],
                    index=["Applied", "To Apply", "Interview", "Rejected", "Offer"].index(app['status']),
                    key=f"status_{idx}"
                )
                if new_status != app['status']:
                    update_application(idx, {'status': new_status})
                    st.rerun()

                # Show materials
                mat1, mat2, mat3 = st.tabs(["📧 Cold Email", "📝 Cover Letter", "✏️ Rewritten CV"])
                with mat1:
                    st.markdown(app.get('cold_email', 'Not saved'))
                with mat2:
                    st.markdown(app.get('cover_letter', 'Not saved'))
                with mat3:
                    st.markdown(app.get('rewritten_cv', 'Not saved'))
