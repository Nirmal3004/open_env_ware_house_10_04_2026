from pydantic import BaseModel, Field
from typing import List, Optional


class PlannerState(BaseModel):
    task_id: str
    difficulty: str
    user_input: str
    goal: str = ""
    draft_plan: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    timeline: str = ""
    feedback: str = ""
    done: bool = False
    step_count: int = 0
    max_steps: int = 5


class PlannerAction(BaseModel):
    action_type: str
    content: Optional[str] = ""


class StepResult(BaseModel):
    observation: PlannerState
    reward: float
    done: bool
    error: Optional[str] = None