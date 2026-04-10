import json
import os
import textwrap
from typing import Any, List, Optional

from my_env.env import WarehouseRobotEnv
from my_env.graders import normalize_score
from my_env.tasks import TASKS as TASK_DEFINITIONS
from openai_client import get_openai_client

from config import API_BASE_URL, API_KEY, MODEL_NAME

BENCHMARK = os.getenv("WAREHOUSE_BENCHMARK", "warehouse_robot_planner")
TASK_FILTER = os.getenv("WAREHOUSE_TASK")
MAX_STEPS = 9
TEMPERATURE = 0.2
MAX_TOKENS = 350
SUCCESS_SCORE_THRESHOLD = 0.1

TASKS = [TASK_FILTER] if TASK_FILTER in TASK_DEFINITIONS else ["easy", "medium", "hard"]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are operating a warehouse management robot planner environment.

    On each turn, return exactly one JSON object with this shape:
    {
      "action_type": "identify_goal | generate_robot_plan | assign_robot | suggest_resources | set_zone_route | add_safety_checks | set_battery_strategy | set_priority | finalize",
      "content": "string or list"
    }

    Rules:
    - Use only the listed action_type values.
    - Choose the next best single action based on the current observation.
    - Keep warehouse planning realistic and safety-aware.
    - If the plan is already complete, return finalize.
    - Return strict JSON only.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    reward = normalize_score(reward)
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    score = normalize_score(score)
    rewards_str = ",".join(f"{normalize_score(r):.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(task_name: str, step: int, observation: dict, last_reward: float, history: List[str]) -> str:
    task = TASK_DEFINITIONS[task_name]
    history_block = "\n".join(history[-5:]) if history else "None"
    return textwrap.dedent(
        f"""
        Task: {task_name}
        Difficulty: {task["difficulty"]}
        Warehouse request: {task["user_input"]}
        Expected goal: {task["expected_goal"]}
        Keywords: {", ".join(task["expected_keywords"])}
        Feedback: {task["feedback"]}

        Step: {step}
        Last reward: {last_reward:.2f}
        Current observation JSON:
        {json.dumps(observation, indent=2)}

        Previous actions:
        {history_block}

        Choose exactly one next action.
        """
    ).strip()


def heuristic_action(observation: dict) -> dict[str, Any]:
    if not observation.get("goal"):
        return {
            "action_type": "identify_goal",
            "content": "Convert the warehouse request into a safe robot execution goal with correct zone movement and completion criteria.",
        }

    if len(observation.get("robot_plan", [])) < 4:
        return {
            "action_type": "generate_robot_plan",
            "content": [
                "Pick the required inventory from the source zone",
                "Move through the planned warehouse route",
                "Scan and place the inventory at the destination",
                "Confirm task completion and inventory update",
            ],
        }

    if not observation.get("assigned_robot"):
        return {"action_type": "assign_robot", "content": "carrier robot with warehouse route support"}

    if not observation.get("tools_or_resources"):
        return {"action_type": "suggest_resources", "content": ["barcode scanner", "cart", "pallet support"]}

    if not observation.get("zone_route"):
        return {
            "action_type": "set_zone_route",
            "content": "Travel through the relevant warehouse zones while avoiding busy aisles and confirming destination access.",
        }

    if len(observation.get("safety_checks", [])) < 2:
        return {
            "action_type": "add_safety_checks",
            "content": ["collision avoidance", "overload check", "restricted zone warning"],
        }

    if not observation.get("battery_strategy"):
        return {
            "action_type": "set_battery_strategy",
            "content": "Assign a robot with sufficient battery and switch to a charged backup robot if battery drops below the safe threshold.",
        }

    if not observation.get("priority_level"):
        return {"action_type": "set_priority", "content": "normal"}

    return {"action_type": "finalize", "content": "Warehouse robot execution plan confirmed."}


def sanitize_action(action: Any, observation: dict) -> dict[str, Any]:
    valid_actions = {
        "identify_goal",
        "generate_robot_plan",
        "assign_robot",
        "suggest_resources",
        "set_zone_route",
        "add_safety_checks",
        "set_battery_strategy",
        "set_priority",
        "finalize",
    }

    if not isinstance(action, dict):
        return heuristic_action(observation)

    action_type = str(action.get("action_type", "")).strip()
    content = action.get("content", "")

    if action_type not in valid_actions:
        return heuristic_action(observation)

    if action_type in {"generate_robot_plan", "suggest_resources", "add_safety_checks"} and not isinstance(content, list):
        return heuristic_action(observation)

    if action_type not in {"generate_robot_plan", "suggest_resources", "add_safety_checks"} and isinstance(content, list):
        content = " ".join(str(item) for item in content)

    return {"action_type": action_type, "content": content}


def get_model_action(task_name: str, step: int, observation: dict, last_reward: float, history: List[str]) -> dict[str, Any]:
    if not API_KEY or not API_BASE_URL:
        return heuristic_action(observation)

    client = get_openai_client()
    user_prompt = build_user_prompt(task_name, step, observation, last_reward, history)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
            response_format={"type": "json_object"},
        )
        text = (completion.choices[0].message.content or "").strip()
        return sanitize_action(json.loads(text), observation)
    except Exception:
        return heuristic_action(observation)


def compute_score(rewards: List[float]) -> float:
    if not rewards:
        return normalize_score(0.5)
    average_reward = sum(normalize_score(value) for value in rewards) / len(rewards)
    return normalize_score(average_reward)


def run_task(task_name: str) -> None:
    env = WarehouseRobotEnv()
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    last_reward = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    state = env.reset(task_name)
    observation = state.model_dump()

    for step in range(1, MAX_STEPS + 1):
        if observation.get("done"):
            break

        action = get_model_action(task_name, step, observation, last_reward, history)
        result = env.step(action)

        reward = normalize_score(result.reward)
        done = result.done
        error = result.error

        rewards.append(reward)
        steps_taken = step
        last_reward = reward
        observation = result.observation.model_dump()

        log_step(step=step, action=action["action_type"], reward=reward, done=done, error=error)
        history.append(f"Step {step}: {json.dumps(action)} -> reward {reward:.2f}")

        if done:
            break

    score = compute_score(rewards)
    success = score >= SUCCESS_SCORE_THRESHOLD
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    for task_name in TASKS:
        run_task(task_name)


if __name__ == "__main__":
    main()
