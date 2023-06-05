from dataclasses import dataclass

@dataclass
class TrainingConfig:
    env_id:str = "LunarLander-v2"
    output_dir: str = "/app/outputs"
    num_iterations: int = 50000
    num_eval_episodes: int = 1
    num_envs: int = 4
    epochs: int = 4
    rollout_length: int = 50
    logging_iterations: int = 500
    learning_rate: float = 5e-4
    gamma: float = 0.99
    clip_epsilon: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    grad_norm: float = 0.5
    seed: int = 2023
    anneal_lr: bool = True
    device: str = "cpu"
