# ---------- Interview ----------
elif ss.phase == "interview":
    user_q = None
    if ss.get("insert_question"):
        with st.form(key="question_form", clear_on_submit=True):
            user_q = st.text_input("Suggested Question", value=ss["insert_question"])
            if st.form_submit_button("Send"):
                ss.chat_logs[cand["id"]].append({"sender": "user", "text": user_q})
                ss["insert_question"] = None
                st.rerun()
    else:
        user_q = st.chat_input("Your question")
        if user_q:
            ss.chat_logs[cand["id"]].append({"sender": "user", "text": user_q})
            result = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt(cand, ss.role_data["role"])},
                    *[
                        {"role": "assistant" if m["sender"] == "ai" else "user", "content": m["text"]}
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
    if st.button("➔ End Interview"):
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
    if st.button("🏋 Submit & Get Feedback", disabled=hire is None):
        ss.hire_id = hire
        ss.phase = "score"
        st.rerun()

# ---------- Score ----------
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
