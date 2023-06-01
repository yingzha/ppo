from dataclasses import dataclass

@dataclass
class TrainingConfig:
    env_id:str = "CartPole-v1"
    output_dir: str = "/app/outputs"
    num_iterations: int = 5000
    num_eval_episodes: int = 10
    num_envs: int = 4
    epochs: int = 4
    rollout_length: int = 50
    logging_iterations: int = 50
    learning_rate: float = 1e-4
    gamma: float = 0.99
    clip_epsilon: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    grad_norm: float = 0.5
    seed: int = 2023
    anneal_lr: bool = True
    capture_video: bool = True
    debug: bool = False
    device: str = "cpu"
