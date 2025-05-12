# interview_trainer.py
# Streamlit Interview‑Training Chatbot  ✦  v0.1
# ----------------------------------------------
# Prereqs:
#   pip install streamlit openai python-dotenv
#   export OPENAI_API_KEY=sk-...

import json, pathlib, re, time, uuid
from collections import defaultdict
from typing import List, Dict

import openai
import streamlit as st
from dotenv import load_dotenv

# ---------- ENV / CONFIG ----------
load_dotenv()                           # optional .env support
openai.api_key  = st.secrets.get("OPENAI_API_KEY") or st.getenv("OPENAI_API_KEY")
DATA_DIR        = pathlib.Path("data")   # folder with *.json files
MODEL_NAME      = "gpt-4o-mini"        # adjust as desired
MAX_CANDIDATES  = 4                    # max interviews per session

# ---------- HELPERS ----------
@st.cache_resource
def load_roles() -> Dict[str, pathlib.Path]:
    """Return mapping of role label -> file path."""
    files = {f.stem.replace("_", " ").title(): f for f in DATA_DIR.glob("*.json")}
    if not files:
        st.error("❗ No JSON files found in candidate_personas/")
    return files

def open_json(path: pathlib.Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def init_session():
    """Initialize session_state keys on first load."""
    ss = st.session_state
    defaults = {
        "phase": "setup",       # setup → shortlist → interview → score
        "role_label": None,
        "role_data": None,      # loaded JSON dict
        "shortlist": [],
        "current_idx": 0,
        "chat_logs": defaultdict(list),  # {candidate_id: [{"sender": "user"|"ai", "text": ...}]}
        "scorecard": {}
    }
    for k, v in defaults.items():
        if k not in ss:
            ss[k] = v

def get_candidate(rank: int, data: dict) -> dict:
    return next(c for c in data["candidates"] if c["rank"] == rank)

def system_prompt(candidate: dict, role: str) -> str:
    resume_lines = [candidate["resume"]["headline"]]
    for exp in candidate["resume"]["experience"]:
        resume_lines.append(f"- {exp['company']} ({exp['years']} yrs): {exp['highlight']}")
    skills = ", ".join(candidate["resume"]["skills"])
    return f"""
You are {candidate['name']}, interviewing for the {role} position.
Answer as the candidate would, based on this résumé:

{chr(10).join(resume_lines)}
Education: {candidate['resume']['education']}
Key skills: {skills}

Be professional, concise, and truthful. Reveal red flags ONLY if the interviewer asks
probing questions (don’t volunteer them unprompted). Keep answers < 3 sentences unless
asked to elaborate."""
# ---------- SCORING ----------
OPEN_PROBE_RE = re.compile(r"^(tell|give|share|describe|walk|can you|what|how)", re.I)

def score_interview(chat: List[dict], candidate: dict) -> Dict[str, int]:
    """Return dict with component + total scores."""
    questions   = [m["text"] for m in chat if m["sender"] == "user"]
    open_qs     = [q for q in questions if OPEN_PROBE_RE.match(q.strip())]
    red_flags   = candidate.get("red_flags", [])
    discovered  = sum(any(flag["label"].lower() in m["text"].lower() for m in chat) for flag in red_flags)

    s_question  = int(30 * len(open_qs) / max(len(questions), 1))
    s_flags     = int(25 * discovered / max(len(red_flags), 1)) if red_flags else 25
    s_flow      = 15  # simple fixed score; could add sentiment/turn‑ratio
    s_etiquette = 10  # placeholder
    subtotal    = s_question + s_flags + s_flow + s_etiquette
    return {
        "question_quality": s_question,
        "red_flag": s_flags,
        "listening_flow": s_flow,
        "etiquette": s_etiquette,
        "subtotal": subtotal
    }

# ---------- UI FLOW ----------
init_session()
roles_map = load_roles()
ss = st.session_state

st.title("🧑‍💼 Interview Training Simulator")

# -------- Phase 1: Setup --------
if ss.phase == "setup":
    ss.role_label = st.selectbox("Select the role you’re hiring for:", list(roles_map))
    if ss.role_label:
        ss.role_data = open_json(roles_map[ss.role_label])
        st.subheader("Résumé stack")
        cols = st.columns(3)
        for c in ss.role_data["candidates"]:
            card = f"**{c['name']}**  \nRank #{c['rank']} — {c['resume']['headline']}"
            cols[(c['rank']-1) % 3].markdown(card)

        st.markdown("---")
        shortlist = st.multiselect("Select up to 4 candidates to interview:",
                                   options=[f"{c['rank']} · {c['name']}" for c in ss.role_data["candidates"]],
                                   max_selections=MAX_CANDIDATES)
        st.caption("Tip: pick a mix of top and mid‑ranked candidates to practice probing skills.")

        if st.button("✅ Start Interviews", disabled=len(shortlist) == 0):
            # store only rank ints
            ss.shortlist = [int(x.split("·")[0].strip()) for x in shortlist]
            ss.phase = "interview"
            st.rerun()

# -------- Phase 2: Interview loop --------
elif ss.phase == "interview":
    current_rank = ss.shortlist[ss.current_idx]
    candidate     = get_candidate(current_rank, ss.role_data)
    st.header(f"Interview {ss.current_idx+1} / {len(ss.shortlist)}  —  {candidate['name']}")

    chat_container = st.container(height=400)

    # render chat history
    for msg in ss.chat_logs[candidate["id"]]:
        align = "user" if msg["sender"] == "user" else "assistant"
        chat_container.chat_message(align).markdown(msg["text"])

    # user input
    prompt = st.chat_input("Your question")
    if prompt:
        ss.chat_logs[candidate["id"]].append({"sender": "user", "text": prompt})

        # GPT reply
        response = openai.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt(candidate, ss.role_data['role'])},
                *[
                    {"role": "assistant" if m["sender"] == "ai" else "user", "content": m["text"]}
                    for m in ss.chat_logs[candidate["id"]]
                    if m["sender"] in ("user", "ai")
                ]
            ],
            max_tokens=150,
            temperature=0.7,
        ).choices[0].message.content.strip()

        ss.chat_logs[candidate["id"]].append({"sender": "ai", "text": response})
        st.rerun()

    st.markdown("---")
    col1, col2 = st.columns([1, 3])
    if col1.button("➜ End Interview"):
        # score this interview
        ss.scorecard[candidate["id"]] = score_interview(ss.chat_logs[candidate["id"]], candidate)
        # next candidate or proceed
        ss.current_idx += 1
        if ss.current_idx >= len(ss.shortlist):
            ss.phase = "selection"
        st.rerun()

# -------- Phase 3: Final selection & overall score --------
elif ss.phase == "selection":
    st.header("Select your final hire")
    candidates_rank_map = {get_candidate(r, ss.role_data)["id"]: r for r in ss.shortlist}
    id_to_name          = {get_candidate(r, ss.role_data)["id"]: get_candidate(r, ss.role_data)["name"]
                           for r in ss.shortlist}
    hire_id = st.selectbox("Who would you hire?", options=list(id_to_name),
                           format_func=lambda cid: id_to_name[cid])

    if st.button("🏁 Submit & Get Feedback", disabled=hire_id is None):
        ss.hire_id = hire_id
        ss.phase = "score"
        st.rerun()

# -------- Phase 4: Score screen --------
elif ss.phase == "score":
    total = 0
    for cid, parts in ss.scorecard.items():
        total += parts["subtotal"]
    # evaluation bonus
    best_rank = min(ss.shortlist)
    chosen_rank = get_candidate(best_rank, ss.role_data)["rank"] if ss.hire_id == get_candidate(best_rank, ss.role_data)["id"] else \
                  next(get_candidate(r, ss.role_data)["rank"] for r in ss.shortlist
                       if get_candidate(r, ss.role_data)["id"] == ss.hire_id)
    eval_score = 20 if chosen_rank == best_rank else (15 if chosen_rank == best_rank + 1 else 5)
    total += eval_score

    st.success(f"🎯 **Overall Interview Score: {total}/100**")
    st.write(f"Evaluation component: {eval_score} pts")

    # strengths / areas
    st.subheader("Strengths")
    st.write("- Asked solid open‑ended questions" if total > 60 else "- Some good questions, but more probing needed")
    st.subheader("Missed Opportunities")
    missed_flags = []
    for rank in ss.shortlist:
        cand = get_candidate(rank, ss.role_data)
        flags = [rf["label"] for rf in cand.get("red_flags", [])]
        if flags:
            flagged = any(flag.lower() in json.dumps(ss.chat_logs[cand["id"]]).lower() for flag in flags)
            if not flagged:
                missed_flags.extend(flags)
    if missed_flags:
        st.write(", ".join(set(missed_flags)))
    else:
        st.write("Great job uncovering the key red flags!")

    if st.button("🔄 Start New Session"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
