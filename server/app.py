from typing import Any

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Query
from pydantic import BaseModel

from my_env.env import WarehouseRobotEnv

app = FastAPI(title="Warehouse Management Robot Planner Environment")
env = WarehouseRobotEnv()


class ResetRequest(BaseModel):
    task_name: str = "easy"


class StepRequest(BaseModel):
    action_type: str
    content: Any = ""


@app.get("/")
def root():
    return {
        "message": "Warehouse Management Robot Planner OpenEnv is running",
        "endpoints": {
            "reset": "POST /reset or GET /reset?task_name=easy",
            "step": "POST /step or GET /step?action_type=identify_goal&content=goal",
            "state": "GET /state",
        },
    }


def _run_reset(task_name: str = "easy"):
    state = env.reset(task_name)
    return state.model_dump()


def _run_step(action_type: str, content: Any = ""):
    action_type = str(action_type).strip()
    if not action_type:
        raise HTTPException(
            status_code=400,
            detail="action_type is required.",
        )

    try:
        result = env.step(
            {
                "action_type": action_type,
                "content": content,
            }
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.model_dump()


@app.get("/reset")
def reset_get(task_name: str = Query(default="easy")):
    return _run_reset(task_name)


@app.get("/restep")
def restep_get(task_name: str = Query(default="easy")):
    return _run_reset(task_name)


@app.post("/reset")
def reset(
    req: ResetRequest | None = Body(
        default=None,
        examples={
            "easy": {"summary": "Easy warehouse task", "value": {"task_name": "easy"}},
            "medium": {"summary": "Medium warehouse task", "value": {"task_name": "medium"}},
            "hard": {"summary": "Hard warehouse task", "value": {"task_name": "hard"}},
        },
    )
):
    task_name = "easy" if req is None else req.task_name
    return _run_reset(task_name)


@app.get("/state")
def get_state():
    if env.state is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first.",
        )
    return env.state_dict()


@app.post("/step")
def step(
    req: dict[str, Any] | None = Body(
        default=None,
        examples={
            "identify_goal": {
                "summary": "Identify the warehouse goal",
                "value": {
                    "action_type": "identify_goal",
                    "content": "Move one inbound package from the receiving zone to rack B2 with scan and safety checks.",
                },
            },
            "generate_robot_plan": {
                "summary": "Create a robot execution plan",
                "value": {
                    "action_type": "generate_robot_plan",
                    "content": [
                        "Scan the inbound package at receiving zone",
                        "Pick the package with the assigned robot",
                        "Travel through the storage route to rack B2",
                        "Place the package and confirm the shelf update",
                    ],
                },
            },
        },
    )
):
    if req is None:
        raise HTTPException(
            status_code=400,
            detail="Request body is required. Provide action_type and content.",
        )

    return _run_step(req.get("action_type", ""), req.get("content", ""))


@app.get("/step")
def step_get(
    action_type: str | None = Query(default=None),
    content: str = Query(default=""),
):
    if action_type is None:
        return {
            "message": "Use POST /step with JSON or GET /step with query parameters.",
            "example": "/step?action_type=identify_goal&content=Move%20package%20to%20rack%20B2",
            "supported_actions": [
                "identify_goal",
                "generate_robot_plan",
                "assign_robot",
                "suggest_resources",
                "set_zone_route",
                "add_safety_checks",
                "set_battery_strategy",
                "set_priority",
                "finalize",
            ],
        }
    return _run_step(action_type, content)


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
