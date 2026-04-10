def _clamp_score(score: float) -> float:
    return round(min(max(score, 0.01), 0.99), 2)


def grade_state(state, task):
    score = 0.01

    goal_text = (state.goal or "").lower()
    plan_text = " ".join(state.robot_plan).lower()
    robot_text = (state.assigned_robot or "").lower()
    resource_text = " ".join(state.tools_or_resources).lower()
    route_text = (state.zone_route or "").lower()
    safety_text = " ".join(state.safety_checks).lower()
    battery_text = (state.battery_strategy or "").lower()
    priority_text = (state.priority_level or "").lower()
    combined = " ".join(
        [
            goal_text,
            plan_text,
            robot_text,
            resource_text,
            route_text,
            safety_text,
            battery_text,
            priority_text,
        ]
    )

    if len(state.goal.strip()) >= 12:
        score += 0.16

    if len(state.robot_plan) >= 4:
        score += 0.2

    if state.assigned_robot.strip():
        score += 0.08

    if state.tools_or_resources:
        score += 0.08

    if any(zone in route_text for zone in ["receiving", "storage", "packing", "dispatch"]):
        score += 0.12

    if len(state.safety_checks) >= 2:
        score += 0.12

    if state.battery_strategy.strip():
        score += 0.09

    if state.priority_level.strip():
        score += 0.05

    hits = 0
    for keyword in task["expected_keywords"]:
        if keyword.lower() in combined:
            hits += 1
    if task["expected_keywords"]:
        score += min(hits / len(task["expected_keywords"]), 1.0) * 0.08

    if state.done:
        score += 0.08

    return _clamp_score(score)
