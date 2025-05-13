# interview_trainer.py — MVP with Ranked Response Behavior + Clickable Questions

import os
import json
import pathlib
import re
from collections import defaultdict

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ---------- CONFIG ----------
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
load_dotenv()
client = OpenAI(
    api_key=st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY")
)

DATA_DIR = pathlib.Path("data")
MODEL_NAME = "gpt-4o"
MAX_CANDIDATES = 4

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
        "phase": "setup",
        "role_label": None,
        "role_data": None,
        "shortlist": [],
        "current_idx": 0,
        "chat_logs": defaultdict(list),
        "scorecard": {},
        "insert_question": None,
    }
    for k, v in defaults.items():
        if k not in ss:
            ss[k] = v

def get_candidate(rank: int, data: dict) -> dict:
    return next(c for c in data["candidates"] if c["rank"] == rank)

def system_prompt(candidate: dict, role: str) -> str:
    rank = candidate["rank"]
    lines = [candidate["resume"]["headline"]]
    for exp in candidate["resume"]["experience"]:
        lines.append(f"- {exp['company']} ({exp['years']} yrs): {exp['highlight']}")
    skills = ", ".join(candidate["resume"]["skills"])

    worst_rank = max(c["rank"] for c in st.session_state.role_data["candidates"])
    if rank == 1:
        quality_instructions = (
            "Give highly polished, thoughtful, articulate responses. "
            "Demonstrate leadership, emotional intelligence, critical thinking, and concise communication."
        )
    elif rank == worst_rank:
        quality_instructions = (
            "Give vague, awkward, or short responses. You may hesitate, over-explain, or deflect blame. "
            "Avoid specifics. Use filler words like 'uh', 'I guess', or 'you know'. "
            "When asked about mistakes, try to shift blame or minimize responsibility."
        )
    else:
        quality_instructions = (
            "Give moderately competent responses — a mix of strong and weak points. "
            "Occasionally use filler words. Miss some key insights. Offer partial or slightly off-topic answers."
        )

    return f"""
You are {candidate['name']}, interviewing for the {role} position.
Answer as the candidate would, based on this résumé:

{chr(10).join(lines)}
Education: {candidate['resume']['education']}
Key skills: {skills}

{quality_instructions}

Be professional unless otherwise specified. Only reveal red flags if the interviewer asks probing questions.
Keep answers under 3 sentences unless asked to elaborate.
"""

OPEN_PROBE_RE = re.compile(r"^(tell|give|share|describe|walk|can you|what|how)", re.I)

def score_interview(chat: list, candidate: dict) -> dict:
    questions = [m["text"] for m in chat if m["sender"] == "user"]
    open_qs = [q for q in questions if OPEN_PROBE_RE.match(q.strip())]
    red_flags = candidate.get("red_flags", [])
    discovered = sum(
        any(flag["label"].lower() in msg["text"].lower() for flag in red_flags)
        for msg in chat
    )
    s_question = int(30 * len(open_qs) / max(len(questions), 1))
    s_flags = int(25 * discovered / max(len(red_flags), 1)) if red_flags else 25
    s_flow = 15
    s_etiquette = 10
    subtotal = s_question + s_flags + s_flow + s_etiquette
    return {
        "question_quality": s_question,
        "red_flag": s_flags,
        "listening_flow": s_flow,
        "etiquette": s_etiquette,
        "subtotal": subtotal
    }

# ---------- SIDEBAR ----------
def show_sidebar_question_guide():
    question_groups = {
        "🧠 Problem-Solving & Critical Thinking": [
            "Tell me about a time you had to solve a difficult problem.",
            "Describe a situation where you had to make a difficult decision.",
            "Give an example of a goal you reached and how you achieved it."
        ],
        "🤝 Teamwork & Collaboration": [
            "Tell me about a time you had to work with a difficult team member.",
            "Describe a situation where you had to collaborate with someone different.",
            "Give an example of a time you had to work under pressure."
        ],
        "🧭 Leadership & Initiative": [
            "Describe a time when you had to demonstrate leadership skills.",
            "Tell me about a time you took initiative on a project.",
            "Give an example of handling a difficult situation and what you learned."
        ],
        "⚖️ Conflict Resolution": [
            "Tell me about a time you had a conflict with a colleague.",
            "Describe a time you disagreed with a superior's decision."
        ],
        "❌ Mistakes & Failures": [
            "Tell me about a time you made a mistake at work.",
            "Describe an occasion when you failed at a task."
        ],
        "📌 Other Common Questions": [
            "What is your biggest strength?",
            "What is your greatest weakness?",
            "Why do you think you're a good fit for this position?",
            "What motivates you?",
            "How do you handle stress?",
            "What are your career goals?"
        ]
    }
    with st.sidebar.expander("💬 Interview Question Guide", expanded=True):
        for group, questions in question_groups.items():
            st.markdown(f"### {group}")
            for q in questions:
                if st.button(q, key=f"q_{q}"):
                    st.session_state.insert_question = q

# ---------- APP ----------
init_session()
ss = st.session_state
roles_map = load_roles()

st.title("🧑‍💼 Interview Training Simulator")
show_sidebar_question_guide()

# ---------- Setup ----------
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

# ---------- Interview ----------
elif ss.phase == "interview":
    rank = ss.shortlist[ss.current_idx]
    cand = get_candidate(rank, ss.role_data)
    st.header(f"Interview {ss.current_idx+1}/{len(ss.shortlist)} — {cand['name']}")

    for msg in ss.chat_logs[cand["id"]]:
        align = "user" if msg["sender"] == "user" else "assistant"
        st.chat_message(align).markdown(msg["text"])

    if ss.get("insert_question"):
        user_q = ss["insert_question"]
        ss.chat_logs[cand["id"]].append({"sender": "user", "text": user_q})
        ss["insert_question"] = None

        result = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt(cand, ss.role_data["role"])}
            ] + [
                {"role": "assistant" if m["sender"] == "ai" else "user", "content": m["text"]}
                for m in ss.chat_logs[cand["id"]]
            ],
            max_tokens=150,
            temperature=0.7,
        )
        resp = result.choices[0].message.content.strip()
        ss.chat_logs[cand["id"]].append({"sender": "ai", "text": resp})
        st.rerun()
    else:
        user_q = st.chat_input("Your question")
        if user_q:
            ss.chat_logs[cand["id"]].append({"sender": "user", "text": user_q})
            result = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt(cand, ss.role_data["role"])}
                ] + [
                    {"role": "assistant" if m["sender"] == "ai" else "user", "content": m["text"]}
                    for m in ss.chat_logs[cand["id"]]
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

# ---------- Selection ----------
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

# ---------- Scoring ----------
elif ss.phase == "score":
    total = sum(v["subtotal"] for v in ss.scorecard.values())
    best = min(ss.shortlist)
    chosen_rank = next(r for r in ss.shortlist if get_candidate(r, ss.role_data)["id"] == ss.hire_id)
    bonus = 20 if chosen_rank == best else (15 if chosen_rank == best + 1 else 5)
    total += bonus

    st.success(f"🎯 **Overall Interview Score: {total}/100** (Bonus: {bonus} pts)")
    st.subheader("Strengths")
    st.write("- Strong open-ended questioning 👍" if total > 60 else "- Work on deeper probing and follow-ups.")
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
