from dataclasses import dataclass

@dataclass
class TrainingConfig:
    env_id:str = "CartPole-v1"
    model_type: str = "lora"
    output_dir: str = "/app/outputs"
    num_iterations: int = 5000
    num_envs: int = 4
    num_steps: int = 50
    epochs: int = 4
    rollout_length: int = 80
    logging_steps: int = 50
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
