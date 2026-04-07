from typing import Any

from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel

from my_env.env import JobReadinessEnv

app = FastAPI(title="Job Readiness Task Planner Environment")
env = JobReadinessEnv()


class ResetRequest(BaseModel):
    task_name: str = "easy"


class StepRequest(BaseModel):
    action_type: str
    content: Any = ""


@app.get("/")
def root():
    return {"message": "Job Readiness OpenEnv is running"}


@app.post("/reset")
def reset(req: ResetRequest):
    state = env.reset(req.task_name)
    return state.model_dump()


@app.get("/state")
def get_state():
    if env.state is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first.",
        )
    return env.state_dict()


@app.post("/step")
def step(req: StepRequest):
    try:
        result = env.step(
            {
                "action_type": req.action_type,
                "content": req.content,
            }
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.model_dump()
