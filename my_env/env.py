import copy
from typing import Any

from my_env.graders import grade_state
from my_env.models import StepResult, WarehouseRobotPlannerState
from my_env.tasks import TASKS


class WarehouseRobotEnv:
    def __init__(self):
        self.current_task = None
        self.state = None

    def reset(self, task_name: str = "easy"):
        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name}")

        task = copy.deepcopy(TASKS[task_name])
        self.current_task = task
        self.state = WarehouseRobotPlannerState(
            task_id=task["task_id"],
            difficulty=task["difficulty"],
            user_input=task["user_input"],
            feedback=task["feedback"],
        )
        return self.state

    def state_dict(self):
        if self.state is None:
            return {"error": "Environment not initialized. Call /reset first."}
        return self.state.model_dump()

    def _normalize_string_list(self, content: Any, field_name: str):
        if not isinstance(content, list):
            raise ValueError(f"{field_name} content must be a list of strings")
        return [str(item).strip() for item in content if str(item).strip()]

    def step(self, action: dict):
        if self.state is None:
            raise ValueError("Environment not initialized. Call reset() first.")

        self.state.step_count += 1
        error = None

        if not isinstance(action, dict):
            error = "Action must be a dictionary with action_type and content."
            reward = grade_state(self.state, self.current_task)
            if self.state.step_count >= self.state.max_steps:
                self.state.done = True
            return StepResult(
                observation=self.state,
                reward=reward,
                done=self.state.done,
                error=error,
            )

        action_type = str(action.get("action_type", "")).strip()
        content = action.get("content", "")

        try:
            if action_type == "identify_goal":
                self.state.goal = str(content).strip()
            elif action_type == "generate_robot_plan":
                self.state.robot_plan = self._normalize_string_list(content, "generate_robot_plan")
            elif action_type == "assign_robot":
                self.state.assigned_robot = str(content).strip()
            elif action_type == "suggest_resources":
                self.state.tools_or_resources = self._normalize_string_list(content, "suggest_resources")
            elif action_type == "set_zone_route":
                self.state.zone_route = str(content).strip()
            elif action_type == "add_safety_checks":
                self.state.safety_checks = self._normalize_string_list(content, "add_safety_checks")
            elif action_type == "set_battery_strategy":
                self.state.battery_strategy = str(content).strip()
            elif action_type == "set_priority":
                self.state.priority_level = str(content).strip()
            elif action_type == "finalize":
                self.state.done = True
            else:
                error = f"Unknown action_type: {action_type}"
        except Exception as exc:
            error = str(exc)

        if self.state.step_count >= self.state.max_steps:
            self.state.done = True

        reward = grade_state(self.state, self.current_task)
        return StepResult(
            observation=self.state,
            reward=reward,
            done=self.state.done,
            error=error,
        )
