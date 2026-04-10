from typing import Any, List, Optional

from pydantic import BaseModel, Field


class WarehouseRobotPlannerState(BaseModel):
    task_id: str
    difficulty: str
    user_input: str
    goal: str = ""
    robot_plan: List[str] = Field(default_factory=list)
    assigned_robot: str = ""
    tools_or_resources: List[str] = Field(default_factory=list)
    zone_route: str = ""
    safety_checks: List[str] = Field(default_factory=list)
    battery_strategy: str = ""
    priority_level: str = ""
    feedback: str = ""
    done: bool = False
    step_count: int = 0
    max_steps: int = 9


class WarehouseRobotPlannerAction(BaseModel):
    action_type: str
    content: Optional[Any] = ""


class StepResult(BaseModel):
    observation: WarehouseRobotPlannerState
    reward: float
    done: bool
    error: Optional[str] = None
