import os
import re
import pdfplumber
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

@st.cache_resource
def load_sbert():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_sbert()

# ===============================
# LOAD SBERT MODEL
# ===============================
model = SentenceTransformer('all-MiniLM-L6-v2')

st.set_page_config(page_title="HR Analytics Dashboard", layout="wide")
st.title("📊 AI Resume Screening & HR Analytics Dashboard")

# ===============================
# JOB ROLES + SKILLS
# ===============================
JOB_DATA = {
    "Data Analyst": {
        "description": """
        Looking for a Data Analyst with experience in Python, SQL,
        Power BI, data visualization, statistics, and machine learning basics.
        """,
        "skills": [
            "python", "sql", "power bi",
            "statistics", "visualization",
            "machine learning"
        ]
    },
    "Machine Learning Engineer": {
        "description": """
        Looking for a Machine Learning Engineer with experience in
        machine learning, deep learning, NLP, TensorFlow, PyTorch,
        model deployment and Python.
        """,
        "skills": [
            "machine learning", "deep learning",
            "nlp", "tensorflow", "pytorch",
            "deployment", "python"
        ]
    },
    "Power BI Analyst": {
        "description": """
        Looking for a Power BI Analyst skilled in Power BI,
        DAX, SQL, dashboards, reporting and business intelligence.
        """,
        "skills": [
            "power bi", "dax", "sql",
            "dashboards", "reporting",
            "business intelligence"
        ]
    }
}

# ===============================
# SIDEBAR
# ===============================
job_role = st.selectbox("🧠 Select Job Role", list(JOB_DATA.keys()))

THRESHOLD = st.slider("🎯 Shortlisting Threshold (%)", 0, 100, 50, 5)

uploaded_files = st.file_uploader(
    "📂 Upload Resume PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# ===============================
# FUNCTIONS
# ===============================
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_skills(resume_text, skill_list):
    matched = []
    for skill in skill_list:
        if skill in resume_text:
            matched.append(skill)
    return matched

# ===============================
# MAIN LOGIC
# ===============================
if st.button("🔍 Analyze Resumes") and uploaded_files:

    resume_texts = []
    resume_names = []
    matched_skills_list = []

    job_description = JOB_DATA[job_role]["description"]
    required_skills = JOB_DATA[job_role]["skills"]

    for file in uploaded_files:
        raw_text = extract_text_from_pdf(file)
        cleaned_text = clean_text(raw_text)

        resume_texts.append(cleaned_text)
        resume_names.append(file.name)

        matched_skills = extract_skills(cleaned_text, required_skills)
        matched_skills_list.append(matched_skills)

    # SBERT Matching
    resume_embeddings = model.encode(resume_texts)
    jd_embedding = model.encode([job_description])

    scores = cosine_similarity(resume_embeddings, jd_embedding).flatten()
    score_percent = (scores * 100).round(2)

    results = pd.DataFrame({
        "Resume": resume_names,
        "Semantic Match Score (%)": score_percent,
        "Matched Skills": [", ".join(s) if s else "None" for s in matched_skills_list]
    }).sort_values(by="Semantic Match Score (%)", ascending=False)

    results["Status"] = results["Semantic Match Score (%)"].apply(
        lambda x: "Shortlisted" if x >= THRESHOLD else "Rejected"
    )

    # ===============================
    # HR METRICS SECTION
    # ===============================
    total = len(results)
    shortlisted_count = len(results[results["Status"] == "Shortlisted"])
    avg_score = round(results["Semantic Match Score (%)"].mean(), 2)
    shortlist_rate = round((shortlisted_count / total) * 100, 2)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Resumes", total)
    col2.metric("Shortlisted", shortlisted_count)
    col3.metric("Shortlist Rate (%)", shortlist_rate)
    col4.metric("Average Match Score", avg_score)

    
    # ===============================
    # RESULTS TABLE
    # ===============================
    st.subheader("🏆 Detailed Resume Ranking")
    st.dataframe(results, use_container_width=True)

else:
    st.info("⬆️ Select job role and upload resumes to start HR analytics.")
    
