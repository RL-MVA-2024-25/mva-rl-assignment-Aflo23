import random
import os
from pathlib import Path
import numpy as np
import torch

from evaluate import evaluate_HIV, evaluate_HIV_population
#from train_ppo_attempt_2 import ProjectAgent  # Replace DummyAgent with your agent implementation
from train_2 import ProjectAgent 

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)

# Define thresholds for the levels
one_env_thresholds = [3432807.680391572, 1e8, 1e9, 1e10, 2e10, 5e10]
dr_env_thresholds = [1e10, 2e10, 5e10]


def calculate_level(one_env_score, dr_env_score):
    """Calculate the final level based on score thresholds."""
    one_env_level = sum(one_env_score >= threshold for threshold in one_env_thresholds)
    dr_env_level = sum(dr_env_score >= threshold for threshold in dr_env_thresholds)
    return one_env_level + dr_env_level


if __name__ == "__main__":
    file = Path("score.txt")
    if not file.is_file():
        seed_everything(seed=42)
        # Initialization of the agent. Replace DummyAgent with your custom agent implementation.
        agent = ProjectAgent()
        agent.load()
        # Evaluate agent and write score.
        score_agent: float = evaluate_HIV(agent=agent, nb_episode=5)
        score_agent_dr: float = evaluate_HIV_population(agent=agent, nb_episode=20)
        # Calculate and print the final level
        level = calculate_level(score_agent, score_agent_dr)
        print(f"Final Level Achieved: {level}")
        with open(file="score.txt", mode="w") as f:
            f.write(f"{score_agent}\n{score_agent_dr}")
