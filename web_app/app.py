import streamlit as st
import PyPDF2
import re
import string
import spacy
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ================== STREAMLIT CONFIG ==================
st.set_page_config(page_title="AI Resume Classifier & Job Matcher")

st.title("📄 AI Resume Classifier & Job Matcher")
st.write("🚀 Section-wise Resume Analysis Enabled")

# ================== LOAD NLP ==================
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
    st.stop()

stop_words = set(stopwords.words("english"))

# ================== TEXT PREPROCESSING ==================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    doc = nlp(text)
    tokens = [t.lemma_ for t in doc if t.text not in stop_words]
    return " ".join(tokens)

# ================== PDF RESUME EXTRACTION ==================
def extract_resume_text(uploaded_file):
    text = ""
    reader = PyPDF2.PdfReader(uploaded_file)
    for page in reader.pages:
        text += page.extract_text()
    return text

# ================== SIMILARITY ==================
def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(vectors[0], vectors[1])[0][0]

# ================== SECTION EXTRACTION ==================
def extract_sections(resume_text):
    sections = {
        "skills": "",
        "experience": "",
        "education": "",
        "projects": ""
    }

    current = None
    for line in resume_text.lower().split("\n"):
        if "skill" in line:
            current = "skills"
        elif "experience" in line:
            current = "experience"
        elif "education" in line:
            current = "education"
        elif "project" in line:
            current = "projects"
        elif current:
            sections[current] += line + " "

    return sections

# ================== SECTION-WISE SCORE ==================
def calculate_section_scores(resume_text, job_description):
    sections = extract_sections(resume_text)
    scores = {}

    for sec, txt in sections.items():
        if txt.strip():
            score = calculate_similarity(
                preprocess_text(txt),
                preprocess_text(job_description)
            )
            scores[sec] = round(score * 100, 2)
        else:
            scores[sec] = 0.0

    return scores

# ================== SKILL GAP ANALYSIS ==================
def extract_skills_from_text(text):
    skills = [
        "python", "sql", "excel", "pandas", "numpy",
        "machine learning", "deep learning",
        "scikit-learn", "tensorflow",
        "aws", "docker", "kubernetes",
        "power bi", "tableau"
    ]
    text = text.lower()
    return set(s for s in skills if s in text)

def find_skill_gap(resume_text, job_description):
    resume_skills = extract_skills_from_text(resume_text)
    job_skills = extract_skills_from_text(job_description)
    return job_skills & resume_skills, job_skills - resume_skills

# ================== AI FEEDBACK ==================
def generate_ai_feedback(section_scores):
    feedback = []
    for sec, score in section_scores.items():
        if score >= 70:
            feedback.append(f"✅ Strong alignment in {sec.capitalize()} section.")
        elif score >= 40:
            feedback.append(f"⚠ {sec.capitalize()} section needs improvement.")
        else:
            feedback.append(f"❌ {sec.capitalize()} section is weak.")
    return feedback

# ================== WEIGHTED SCORE ==================
def weighted_score(scores):
    weights = {
        "skills": 0.4,
        "experience": 0.3,
        "projects": 0.2,
        "education": 0.1
    }
    return round(sum(scores[s] * w for s, w in weights.items()), 2)

# ================== ROLE CLASSIFICATION ==================
def classify_resume(resume_text):
    t = resume_text.lower()
    if "data" in t:
        return "Data Analyst"
    elif "devops" in t:
        return "DevOps Engineer"
    elif "developer" in t:
        return "Software Developer"
    else:
        return "General Profile"

# ================== ATS SCORE ==================
def ats_score(resume_text):
    length_score = min(len(resume_text.split()) / 500, 1) * 40
    keyword_score = len(extract_skills_from_text(resume_text)) * 5
    return round(min(length_score + keyword_score, 100), 2)

# ================== WEB UI ==================
resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description")

if st.button("Match Resume"):
    if resume_file and job_description:
        resume_text = extract_resume_text(resume_file)

        # Section scores
        section_scores = calculate_section_scores(resume_text, job_description)

        st.subheader("📊 Section-wise Match Score")
        for sec, sc in section_scores.items():
            st.write(f"**{sec.capitalize()} Match:** {sc}%")

        # Overall scores
        overall = sum(section_scores.values()) / len(section_scores)
        st.success(f"✅ Overall Match Score: {round(overall, 2)}%")
        st.success(f"🎯 Weighted Match Score: {weighted_score(section_scores)}%")

        # Visualization
        df = pd.DataFrame(section_scores.items(), columns=["Section", "Score"])
        st.subheader("📊 Visual Analysis")
        st.bar_chart(df.set_index("Section"))

        # Skill gap
        matched, missing = find_skill_gap(resume_text, job_description)
        st.subheader("🧠 Skill Gap Analysis")
        st.write("✅ Matched Skills:", ", ".join(matched) if matched else "None")
        st.write("❌ Missing Skills:", ", ".join(missing) if missing else "None")

        # AI feedback
        st.subheader("🤖 AI Resume Feedback")
        for line in generate_ai_feedback(section_scores):
            st.write(line)

        # Role & ATS
        st.info(f"🧩 Predicted Resume Role: {classify_resume(resume_text)}")
        st.info(f"📄 ATS Compatibility Score: {ats_score(resume_text)}%")

    else:
        st.warning("⚠️ Please upload resume and job description")
