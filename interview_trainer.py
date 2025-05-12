# interview_trainer.py
# Streamlit Interview‑Training Chatbot  ✦  v0.3
# ----------------------------------------------
# Prereqs:
#   pip install streamlit openai python-dotenv
#   (Set OPENAI_API_KEY in Streamlit secrets or via os env)

import os
import json
import pathlib
import re
import uuid
from collections import defaultdict

import openai
import streamlit as st
from dotenv import load_dotenv

# ---------- ENV / CONFIG ----------
load_dotenv()  # optional: pull from .env locally
openai.api_key = st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY")

DATA_DIR       = pathlib.Path("data")  # your folder with sales_representative.json, etc.
MODEL_NAME     = "gpt-4o-mini"         # or gpt-4o, gpt-4-turbo, etc.
MAX_CANDIDATES = 4                     # max interviews per session

# ---------- HELPERS ----------
@st.cache_resource
def load_roles() -> dict:
    files = {f.stem.replace("_", " ").title(): f for f in DATA_DIR.glob("*.json")}
    if not files:
        st.error(f"❗ No JSON files found in {DATA_DIR}/")
    return files

def open_json(path: pathlib.Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def init_session():
    ss = st.session_state
    defaults = {
        "phase":       "setup",       # setup → shortlist → interview → selection → score
        "role_label":  None,
        "role_data":   None,
        "shortlist":   [],
        "current_idx": 0,
        "chat_logs":   defaultdict(list),
        "scorecard":   {},
    }
    for k, v in defaults.items():
        if k not in ss:
            ss[k] = v

def get_candidate(rank: int, data: dict) -> dict:
    return next(c for c in data["candidates"] if c["rank"] == rank)

def system_prompt(candidate: dict, role: str) -> str:
    lines = [candidate["resume"]["headline"]]
    for exp in candidate["resume"]["experience"]:
        lines.append(f"- {exp['company']} ({exp['years']} yrs): {exp['highlight']}")
    skills = ", ".join(candidate["resume"]["skills"])
    return f"""
You are {candidate['name']}, interviewing for the {role} position.
Answer as the candidate would, based on this résumé:

{chr(10).join(lines)}
Education: {candidate['resume']['education']}
Key skills: {skills}

Be professional, concise, and truthful. Reveal red flags ONLY if asked probing questions.
Keep answers under 3 sentences unless asked to elaborate.
"""

# ---------- SCORING ----------
OPEN_PROBE_RE = re.compile(r"^(tell|give|share|describe|walk|can you|what|how)", re.I)

def score_interview(chat: list, candidate: dict) -> dict:
    questions = [m["text"] for m in chat if m["sender"] == "user"]
    open_qs   = [q for q in questions if OPEN_PROBE_RE.match(q.strip())]
    red_flags = candidate.get("red_flags", [])
    discovered = sum(
        any(flag["label"].lower() in msg["text"].lower() for flag in red_flags)
        for msg in chat
    )

    s_question  = int(30 * len(open_qs) / max(len(questions), 1))
    s_flags     = int(25 * discovered / max(len(red_flags), 1)) if red_flags else 25
    s_flow      = 15  # placeholder for flow/listening
    s_etiquette = 10  # placeholder for professionalism
    subtotal    = s_question + s_flags + s_flow + s_etiquette

    return {
        "question_quality": s_question,
        "red_flag":         s_flags,
        "listening_flow":   s_flow,
        "etiquette":        s_etiquette,
        "subtotal":         subtotal
    }

# ---------- APP FLOW ----------
init_session()
ss = st.session_state
roles_map = load_roles()

st.title("🧑‍💼 Interview Training Simulator")

# -- Phase 1: Setup --
if ss.phase == "setup":
    ss.role_label = st.selectbox("Select the role you’re hiring for:", list(roles_map))
    if ss.role_label:
        ss.role_data = open_json(roles_map[ss.role_label])

        st.subheader("Résumé Stack")
        cols = st.columns(3)
        for c in ss.role_data["candidates"]:
            cols[(c["rank"]-1) % 3].markdown(
                f"**{c['name']}**  \nRank #{c['rank']} — {c['resume']['headline']}"
            )

        st.markdown("---")
        shortlist = st.multiselect(
            "Select up to 4 candidates to interview:",
            options=[f"{c['rank']} · {c['name']}" for c in ss.role_data["candidates"]],
            max_selections=MAX_CANDIDATES
        )
        if st.button("✅ Start Interviews", disabled=len(shortlist)==0):
            ss.shortlist = [int(x.split("·")[0].strip()) for x in shortlist]
            ss
