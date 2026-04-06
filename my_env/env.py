import copy
from my_env.models import PlannerState, StepResult
from my_env.tasks import TASKS
from my_env.graders import grade_state


class JobReadinessEnv:
    def __init__(self):
        self.current_task = None
        self.state = None

    def reset(self, task_name: str = "easy"):
        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name}")

        task = TASKS[task_name]
        self.current_task = copy.deepcopy(task)

        self.state = PlannerState(
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

    def step(self, action: dict):
        if self.state is None:
            raise ValueError("Environment not initialized. Call reset() first.")

        self.state.step_count += 1
        error = None

        action_type = action.get("action_type", "")
        content = action.get("content", "")

        try:
            if action_type == "identify_goal":
                self.state.goal = str(content).strip()

            elif action_type == "generate_plan":
                if isinstance(content, list):
                    self.state.draft_plan = [str(x) for x in content]
                else:
                    error = "generate_plan content must be a list"

            elif action_type == "suggest_tools":
                if isinstance(content, list):
                    self.state.tools = [str(x) for x in content]
                else:
                    error = "suggest_tools content must be a list"

            elif action_type == "set_timeline":
                self.state.timeline = str(content).strip()

            elif action_type == "finalize":
                self.state.done = True

            else:
                error = f"Unknown action_type: {action_type}"

        except Exception as e:
            error = str(e)

        reward = grade_state(self.state, self.current_task)

        if self.state.step_count >= self.state.max_steps:
            self.state.done = True

        return StepResult(
            observation=self.state,
            reward=reward,
            done=self.state.done,
            error=error
        )