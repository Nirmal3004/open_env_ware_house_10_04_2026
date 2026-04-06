from fastapi import FastAPI
from pydantic import BaseModel
from my_env.env import JobReadinessEnv

app = FastAPI(title="Job Readiness Task Planner Environment")
env = JobReadinessEnv()


class ResetRequest(BaseModel):
    task_name: str = "easy"


class StepRequest(BaseModel):
    action_type: str
    content: object = ""


@app.get("/")
def root():
    return {"message": "Job Readiness OpenEnv is running"}


@app.post("/reset")
def reset(req: ResetRequest):
    state = env.reset(req.task_name)
    return state.model_dump()


@app.get("/state")
def get_state():
    return env.state_dict()


@app.post("/step")
def step(req: StepRequest):
    result = env.step({
        "action_type": req.action_type,
        "content": req.content
    })
    return result.model_dump()