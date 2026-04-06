def grade_state(state, task):
    score = 0.0

    goal_text = (state.goal or "").lower()
    plan_text = " ".join(state.draft_plan).lower()
    tools_text = " ".join(state.tools).lower()
    timeline_text = (state.timeline or "").lower()
    combined = " ".join([goal_text, plan_text, tools_text, timeline_text])

    # Goal quality
    if state.goal and len(state.goal.strip()) > 5:
        score += 0.2

    # Plan quality
    if len(state.draft_plan) >= 3:
        score += 0.3

    # Relevant keywords
    hits = 0
    for kw in task["expected_keywords"]:
        if kw.lower() in combined:
            hits += 1
    score += min(hits / len(task["expected_keywords"]), 1.0) * 0.3

    # Timeline
    if state.timeline.strip():
        score += 0.1

    # Finalized
    if state.done:
        score += 0.1

    return round(min(score, 1.0), 2)