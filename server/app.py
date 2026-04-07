from fastapi import FastAPI
import uvicorn
from my_env.env import JobReadinessEnv
from pydantic import BaseModel

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


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()