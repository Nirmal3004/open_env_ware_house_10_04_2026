import requests
import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "rule_based_baseline")
TASKS = ["easy", "medium", "hard"]


def log_start(task):
    print(f"[START] task={task} env=job_readiness_env model={MODEL_NAME}", flush=True)


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


def post(path, payload):
    url = f"{API_BASE_URL}{path}"
    return requests.post(url, json=payload, timeout=30).json()


def run_task(task_name):
    log_start(task_name)
    rewards = []

    state = post("/reset", {"task_name": task_name})

    steps = [
        {
            "action_type": "identify_goal",
            "content": "Learn AI tools and skills for job readiness"
        },
        {
            "action_type": "generate_plan",
            "content": [
                "Learn Python and AI basics",
                "Practice SQL, data handling, and beginner machine learning concepts",
                "Build 2 small projects using AI tools",
                "Create resume and portfolio"
            ]
        },
        {
            "action_type": "suggest_tools",
            "content": ["Python", "ChatGPT", "Hugging Face", "GitHub"]
        },
        {
            "action_type": "set_timeline",
            "content": "4-month beginner-friendly roadmap with weekly goals"
        },
        {
            "action_type": "finalize",
            "content": "done"
        },
    ]

    success = False
    step_num = 0

    for action in steps:
        step_num += 1
        result = post("/step", action)
        reward = result["reward"]
        done = result["done"]
        error = result.get("error")
        rewards.append(reward)

        log_step(step_num, action["action_type"], reward, done, error)

        if done:
            success = reward >= 0.7
            break

    log_end(success, step_num, rewards)


if __name__ == "__main__":
    for task in TASKS:
        run_task(task)