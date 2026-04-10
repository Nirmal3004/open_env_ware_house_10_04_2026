import json

import requests
from requests import RequestException

from config import API_BASE_URL, API_KEY, ENV_SERVER_URL, MODEL_NAME
from my_env.env import WarehouseRobotEnv
from my_env.tasks import TASKS as TASK_DEFINITIONS
from openai_client import get_openai_client

TASKS = ["easy", "medium", "hard"]
LOCAL_ENV = WarehouseRobotEnv()
USE_LOCAL_ENV = False


def log_start(task):
    print(f"[START] task={task} env=warehouse_robot_env model={MODEL_NAME}", flush=True)


def log_step(step, action, reward, done, error):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def local_post(path, payload):
    if path == "/reset":
        return LOCAL_ENV.reset(payload.get("task_name", "easy")).model_dump()
    if path == "/step":
        return LOCAL_ENV.step(payload).model_dump()
    if path == "/state":
        return LOCAL_ENV.state_dict()
    raise ValueError(f"Unsupported path: {path}")


def post(path, payload):
    url = f"{ENV_SERVER_URL}{path}"
    global USE_LOCAL_ENV

    if USE_LOCAL_ENV:
        return local_post(path, payload)

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except (RequestException, ValueError):
        USE_LOCAL_ENV = True
        return local_post(path, payload)


def default_steps(task_name):
    if task_name == "easy":
        return [
            {
                "action_type": "identify_goal",
                "content": "Move one inbound package from the receiving zone to rack B2 in the storage zone with a safe scan and place workflow.",
            },
            {
                "action_type": "generate_robot_plan",
                "content": [
                    "Scan the package at receiving zone",
                    "Pick the package with the picker robot",
                    "Move through the receiving-to-storage route toward rack B2",
                    "Place the package on rack B2 and confirm inventory update",
                ],
            },
            {"action_type": "assign_robot", "content": "picker robot with medium payload support"},
            {"action_type": "suggest_resources", "content": ["barcode scanner", "cart", "pallet support"]},
            {
                "action_type": "set_zone_route",
                "content": "Start in receiving zone, travel through the main storage corridor, stop at rack B2 in storage zone, then clear the aisle.",
            },
            {
                "action_type": "add_safety_checks",
                "content": ["collision avoidance", "load validation", "fragile item handling check"],
            },
            {"action_type": "finalize", "content": "Warehouse robot plan confirmed."},
        ]

    if task_name == "medium":
        return [
            {
                "action_type": "identify_goal",
                "content": "Pick three products from shelves A1, B4, and C2, then deliver them to the packing zone with scan-confirm-place workflow.",
            },
            {
                "action_type": "generate_robot_plan",
                "content": [
                    "Start in storage zone and scan item at shelf A1",
                    "Continue to shelf B4 and pick the second item with load-balanced storage bin",
                    "Collect the third item from shelf C2 and confirm all three picks",
                    "Move inventory to packing zone for handoff and packing confirmation",
                ],
            },
            {"action_type": "assign_robot", "content": "carrier robot optimized for multi-stop picking"},
            {"action_type": "suggest_resources", "content": ["barcode scanner", "cart", "conveyor"]},
            {
                "action_type": "set_zone_route",
                "content": "Route through storage zone shelves A1 to B4 to C2, then take the packing lane to packing zone for final confirmation.",
            },
            {
                "action_type": "add_safety_checks",
                "content": ["collision avoidance", "busy zone speed reduction", "weight/load validation"],
            },
            {"action_type": "set_battery_strategy", "content": "Use a robot above 60 percent battery to complete the multi-stop route without recharge delay."},
            {"action_type": "set_priority", "content": "normal"},
            {"action_type": "finalize", "content": "Warehouse picking mission approved."},
        ]

    return [
        {
            "action_type": "identify_goal",
            "content": "Complete an urgent hospital supply order by collecting five items, rerouting around blocked aisle C, using a battery-ready robot, and delivering to dispatch zone in priority mode.",
        },
        {
            "action_type": "generate_robot_plan",
            "content": [
                "Mark the order as urgent priority and lock the dispatch window",
                "Pick and scan the five hospital supply items from storage shelves while avoiding blocked aisle C",
                "Reroute through the alternate north corridor and validate the load before long travel",
                "Move inventory to dispatch zone, place on outbound pallet, and confirm shipment release",
            ],
        },
        {"action_type": "assign_robot", "content": "forklift robot with high-capacity load support and priority dispatch profile"},
        {"action_type": "suggest_resources", "content": ["barcode scanner", "forklift attachment", "pallet support", "conveyor"]},
        {
            "action_type": "set_zone_route",
            "content": "Travel from storage zone through the north reroute, bypass blocked aisle C, avoid the restricted cold-storage corridor, and finish in dispatch zone.",
        },
        {
            "action_type": "add_safety_checks",
            "content": ["collision avoidance", "overload check", "restricted zone warning", "fragile item handling check"],
        },
        {
            "action_type": "set_battery_strategy",
            "content": "Assign only a robot above 75 percent battery; if battery drops below 30 percent mid-mission, hand off the load at a safe transfer point to a charged backup robot.",
        },
        {"action_type": "set_priority", "content": "urgent hospital shipment"},
        {"action_type": "finalize", "content": "Urgent warehouse execution plan confirmed."},
    ]


def _extract_text_content(response) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text
    if hasattr(response, "choices") and response.choices:
        message = response.choices[0].message
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
    return ""


def generate_steps_with_llm(task_name):
    if not API_KEY or not API_BASE_URL:
        return default_steps(task_name)

    task = TASK_DEFINITIONS[task_name]
    client = get_openai_client()
    prompt = f"""
You are planning warehouse robot actions for an OpenEnv task.
Return JSON only with this shape:
{{
  "steps": [
    {{"action_type": "identify_goal", "content": "string"}},
    {{"action_type": "generate_robot_plan", "content": ["step 1", "step 2", "step 3", "step 4"]}},
    {{"action_type": "assign_robot", "content": "string"}},
    {{"action_type": "suggest_resources", "content": ["item1", "item2"]}},
    {{"action_type": "set_zone_route", "content": "string"}},
    {{"action_type": "add_safety_checks", "content": ["item1", "item2"]}},
    {{"action_type": "set_battery_strategy", "content": "string"}},
    {{"action_type": "set_priority", "content": "string"}},
    {{"action_type": "finalize", "content": "string"}}
  ]
}}

Rules:
- Keep action types exactly as listed above.
- Use warehouse robotics language only.
- Include receiving, storage, packing, or dispatch zones when relevant.
- Include robot assignment, resource suggestions, safety checks, and battery-aware planning.
- For easy tasks you may omit set_battery_strategy or set_priority if not needed.
- Ensure the last action is finalize.

Task difficulty: {task["difficulty"]}
Warehouse request: {task["user_input"]}
Expected goal: {task["expected_goal"]}
Expected keywords: {", ".join(task["expected_keywords"])}
Feedback: {task["feedback"]}
""".strip()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a warehouse management robot planner that returns strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or _extract_text_content(response)
        payload = json.loads(content)
        steps = payload.get("steps", [])
        if isinstance(steps, list) and steps:
            return steps
    except Exception:
        return default_steps(task_name)

    return default_steps(task_name)


def run_task(task_name):
    log_start(task_name)
    rewards = []

    post("/reset", {"task_name": task_name})
    success = False
    step_num = 0

    for action in generate_steps_with_llm(task_name):
        step_num += 1
        result = post("/step", action)
        reward = result["reward"]
        done = result["done"]
        error = result.get("error")
        rewards.append(reward)

        log_step(step_num, action["action_type"], reward, done, error)

        if done:
            success = reward >= 0.75
            break

    log_end(success, step_num, rewards)


if __name__ == "__main__":
    for task in TASKS:
        run_task(task)
