import asyncio
import json
import os
import textwrap
from typing import Any, List, Optional

from openai import OpenAI

from config import API_BASE_URL, API_KEY, MODEL_NAME
from my_env.env import WarehouseRobotEnv
from my_env.graders import normalize_score
from my_env.models import WarehouseRobotPlannerAction
from my_env.tasks import TASKS as TASK_DEFINITIONS

IMAGE_NAME = os.getenv("IMAGE_NAME")
TASK_NAME = os.getenv("WAREHOUSE_TASK") or os.getenv("MY_ENV_V4_TASK")
BENCHMARK = os.getenv("WAREHOUSE_BENCHMARK") or os.getenv("MY_ENV_V4_BENCHMARK") or "warehouse_robot_planner"

MAX_STEPS = 9
TEMPERATURE = 0.2
MAX_TOKENS = 350
SUCCESS_SCORE_THRESHOLD = 0.1

MAX_TOTAL_REWARD = MAX_STEPS * 0.99

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are interacting with a warehouse management robot planner environment.

    On each turn, you must return exactly one JSON object with this shape:
    {
      "action_type": "identify_goal | generate_robot_plan | assign_robot | suggest_resources | set_zone_route | add_safety_checks | set_battery_strategy | set_priority | finalize",
      "content": "string or list"
    }

    Your goal is to maximize planning quality by building a complete, safe, warehouse robot execution plan.
    Use warehouse robotics language only.
    Return JSON only.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    reward = normalize_score(reward)
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


def build_user_prompt(
    task_name: str,
    step: int,
    observation: dict[str, Any],
    last_reward: float,
    history: List[str],
) -> str:
    task = TASK_DEFINITIONS[task_name]
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Last reward: {last_reward:.2f}
        Warehouse request: {task["user_input"]}
        Expected goal: {task["expected_goal"]}
        Expected keywords: {", ".join(task["expected_keywords"])}
        Feedback: {task["feedback"]}

        Current observation:
        {json.dumps(observation, indent=2)}

        Previous steps:
        {history_block}

        Send your next action as one JSON object.
        """
    ).strip()


def heuristic_action(observation: dict[str, Any]) -> WarehouseRobotPlannerAction:
    if not observation.get("goal"):
        return WarehouseRobotPlannerAction(
            action_type="identify_goal",
            content="Translate the warehouse request into a clear robot execution goal with source zone, destination zone, and completion criteria.",
        )

    if len(observation.get("robot_plan", [])) < 4:
        return WarehouseRobotPlannerAction(
            action_type="generate_robot_plan",
            content=[
                "Pick inventory from the source location",
                "Move through the planned warehouse route",
                "Scan and place inventory at the target location",
                "Confirm task completion and inventory update",
            ],
        )

    if not observation.get("assigned_robot"):
        return WarehouseRobotPlannerAction(
            action_type="assign_robot",
            content="carrier robot with route-aware warehouse navigation",
        )

    if not observation.get("tools_or_resources"):
        return WarehouseRobotPlannerAction(
            action_type="suggest_resources",
            content=["barcode scanner", "cart", "pallet support"],
        )

    if not observation.get("zone_route"):
        return WarehouseRobotPlannerAction(
            action_type="set_zone_route",
            content="Move through the required warehouse zones while avoiding blocked aisles and busy corridors.",
        )

    if len(observation.get("safety_checks", [])) < 2:
        return WarehouseRobotPlannerAction(
            action_type="add_safety_checks",
            content=["collision avoidance", "overload check", "restricted zone warning"],
        )

    if not observation.get("battery_strategy"):
        return WarehouseRobotPlannerAction(
            action_type="set_battery_strategy",
            content="Assign a sufficiently charged robot and switch to a backup robot if battery drops below the safe threshold.",
        )

    if not observation.get("priority_level"):
        return WarehouseRobotPlannerAction(action_type="set_priority", content="normal")

    return WarehouseRobotPlannerAction(action_type="finalize", content="Warehouse robot execution plan confirmed.")


def sanitize_action(action: Any, observation: dict[str, Any]) -> WarehouseRobotPlannerAction:
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

    if action_type in {"generate_robot_plan", "suggest_resources", "add_safety_checks"}:
        if not isinstance(content, list):
            return heuristic_action(observation)
    elif isinstance(content, list):
        content = " ".join(str(item) for item in content)

    return WarehouseRobotPlannerAction(action_type=action_type, content=content)


def get_model_action(
    client: OpenAI,
    task_name: str,
    step: int,
    observation: dict[str, Any],
    last_reward: float,
    history: List[str],
) -> WarehouseRobotPlannerAction:
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
        if not text:
            return heuristic_action(observation)
        return sanitize_action(json.loads(text), observation)
    except Exception:
        return heuristic_action(observation)


def compute_score(rewards: List[float]) -> float:
    total_reward = sum(normalize_score(reward) for reward in rewards)
    raw_score = total_reward / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.5
    return normalize_score(raw_score)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    task_names = [TASK_NAME] if TASK_NAME in TASK_DEFINITIONS else ["easy", "medium", "hard"]

    for task_name in task_names:
        env = WarehouseRobotEnv()
        history: List[str] = []
        rewards: List[float] = []
        steps_taken = 0
        score = 0.0
        success = False

        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        try:
            state = env.reset(task_name)
            observation = state.model_dump()
            last_reward = 0.0

            for step in range(1, MAX_STEPS + 1):
                if observation.get("done"):
                    break

                action = get_model_action(client, task_name, step, observation, last_reward, history)
                result = env.step(action.model_dump())

                reward = normalize_score(result.reward)
                done = result.done
                error = result.error

                rewards.append(reward)
                steps_taken = step
                observation = result.observation.model_dump()
                last_reward = reward

                log_step(step=step, action=action.action_type, reward=reward, done=done, error=error)
                history.append(f"Step {step}: {action.model_dump_json()} -> reward {reward:+.2f}")

                if done:
                    break

            score = compute_score(rewards)
            success = score >= SUCCESS_SCORE_THRESHOLD

        finally:
            await asyncio.sleep(0)
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
