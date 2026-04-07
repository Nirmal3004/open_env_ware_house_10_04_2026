from fastapi import FastAPI
from my_env.env import JobReadinessEnv

app = FastAPI()
env = JobReadinessEnv()

@app.get("/")
def root():
    return {"message": "OpenEnv running"}

@app.post("/reset")
def reset():
    return env.reset("easy").model_dump()

@app.get("/state")
def state():
    return env.state_dict()

@app.post("/step")
def step():
    return {"message": "step working"}