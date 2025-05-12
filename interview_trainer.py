# interview_trainer.py
# Streamlit Interview Training Chatbot ✦ v0.3.1

import os
import json
import pathlib
import re
from collections import defaultdict

import openai
import streamlit as st
from dotenv import load_dotenv

# ---------- ENV / CONFIG ----------
load_dotenv()
openai.api_key = st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY")

DATA_DIR = pathlib.Path("data")       # folder with your candidate JSON files
MODEL_NAME = "gpt-4o-mini"            # or gpt-4o, gpt-4-turbo, etc.
MAX_CANDIDATES = 4                    # number of candidates user can interview

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
        "phase": "setup",       # setup → interview → selection → score
        "role_label": None,
        "role_data": None,
        "shortlist": [],
        "current_idx": 0,
        "chat_logs": defaultdict(list),
        "scorecard": {},
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
    open_qs = [q for q in questions if OPEN_PROBE_RE.match(q.strip())]
    red_flags = candidate.get("red_flags", [])
    discovered = sum(
        any(flag["label"].lower() in msg["text"].lower() for flag in red_flags)
        for msg in chat
    )

    s_question  = int(30 * len(open_qs) / max(len(questions), 1))
    s_flags     = int(25 * discovered / max(len(red_flags), 1)) if red_flags else 25
    s_flow      = 15  # placeholder
    s_etiquette = 10  # placeholder
    subtotal    = s_question + s_flags + s_flow + s_etiquette

    return {
        "question_quality": s_question,
        "red_flag": s_flags,
        "listening_flow": s_flow,
        "etiquette": s_etiquette,
        "subtotal": subtotal
    }

# ---------- MAIN APP FLOW ----------
init_session()
ss = st.session_state
roles_map = load_roles()

st.title("🧑‍💼 Interview Training Simulator")

# ---------- Phase 1: Setup ----------
if ss.phase == "setup":
    ss.role_label = st.selectbox("Select the role you’re hiring for:", list(roles_map))
    if ss.role_label:
        ss.role_data = open_json(roles_map[ss.role_label])

        st.subheader("Résumé Stack")
        cols = st.columns(3)
        for c in ss.role_data["candidates"]:
            cols[(c["rank"] - 1) % 3].markdown(
                f"**{c['name']}**  \nRank #{c['rank']} — {c['resume']['headline']}"
            )

        st.markdown("---")
        shortlist = st.multiselect(
            "Select up to 4 candidates to interview:",
            options=[f"{c['rank']} · {c['name']}" for c in ss.role_data["candidates"]],
            max_selections=MAX_CANDIDATES
        )
        if st.button("✅ Start Interviews", disabled=len(shortlist) == 0):
            ss.shortlist = [int(x.split("·")[0].strip()) for x in shortlist]
            ss.phase = "interview"
            st.rerun()

# ---------- Phase 2: Interview ----------
elif ss.phase == "interview":
    rank = ss.shortlist[ss.current_idx]
    cand = get_candidate(rank, ss.role_data)
    st.header(f"Interview {ss.current_idx+1}/{len(ss.shortlist)} — {cand['name']}")

    for msg in ss.chat_logs[cand["id"]]:
        align = "user" if msg["sender"] == "user" else "assistant"
        st.chat_message(align).markdown(msg["text"])

    user_q = st.chat_input("Your question")
    if user_q:
        ss.chat_logs[cand["id"]].append({"sender": "user", "text": user_q})
        result = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt(cand, ss.role_data["role"])},
                *[
                    {"role": ("assistant" if m["sender"] == "ai" else "user"), "content": m["text"]}
                    for m in ss.chat_logs[cand["id"]]
                ]
            ],
            max_tokens=150,
            temperature=0.7,
        )
        resp = result.choices[0].message.content.strip()
        ss.chat_logs[cand["id"]].append({"sender": "ai", "text": resp})
        st.rerun()

    st.markdown("---")
    if st.button("➜ End Interview"):
        ss.scorecard[cand["id"]] = score_interview(ss.chat_logs[cand["id"]], cand)
        ss.current_idx += 1
        if ss.current_idx >= len(ss.shortlist):
            ss.phase = "selection"
        st.rerun()

# ---------- Phase 3: Candidate Selection ----------
elif ss.phase == "selection":
    st.header("Select your final hire")
    id_map = {
        get_candidate(r, ss.role_data)["id"]: get_candidate(r, ss.role_data)["name"]
        for r in ss.shortlist
    }
    hire = st.selectbox("Who would you hire?", options=list(id_map), format_func=lambda cid: id_map[cid])
    if st.button("🏁 Submit & Get Feedback", disabled=hire is None):
        ss.hire_id = hire
        ss.phase = "score"
        st.rerun()

# ---------- Phase 4: Scoring ----------
elif ss.phase == "score":
    total = sum(v["subtotal"] for v in ss.scorecard.values())
    best = min(ss.shortlist)
    chosen_rank = next(r for r in ss.shortlist if get_candidate(r, ss.role_data)["id"] == ss.hire_id)
    bonus = 20 if chosen_rank == best else (15 if chosen_rank == best + 1 else 5)
    total += bonus

    st.success(f"🎯 **Overall Interview Score: {total}/100** (Evaluation Bonus: {bonus} pts)")

    st.subheader("Strengths")
    st.write("- Strong open-ended questioning 👍" if total > 60 else "- Good start, but dig deeper with probing questions.")

    st.subheader("Missed Opportunities")
    missed = []
    for r in ss.shortlist:
        cand = get_candidate(r, ss.role_data)
        for f in cand.get("red_flags", []):
            if f["label"].lower() not in json.dumps(ss.chat_logs[cand["id"]]).lower():
                missed.append(f["label"])
    st.write(", ".join(set(missed)) or "None — great job uncovering red flags!")

    if st.button("🔄 Start New Session"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
