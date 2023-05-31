import torch
import torch.nn as nn
import gymnasium as gym


def make_vec_env(env_id: str, num_env: int, seed: int, start_index: int = 0,
                 output_dir: str = "./", capture_video: bool = False):
    def _make_env(rank, env_idx):
        env = gym.make(env_id)
        env = gym.wrappers.record_episode_statistics.RecordEpisodeStatistics(env)
        if capture_video and env_idx == 0:
            run_name = env_id + "_" + str(env_idx)
            env = gym.wrappers.RecordVideo(env, f"{output_dir}/videos/{run_name}")
        # seed everything
        env.reset(seed = rank + seed)
        env.action_space.seed(rank + seed)
        env.observation_space.seed(rank + seed)
        return env

    envs = gym.vector.SyncVectorEnv(
        [lambda: _make_env(i + start_index, i) for i in range(num_env)])

    return envs


def init_tensors(num_steps, num_envs, obs_space_shape, action_space_shape, device = "cpu"):
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"{device} is invalid!")

    observations = torch.zeros((num_steps, num_envs) + obs_space_shape, device=device)
    actions = torch.zeros((num_steps, num_envs) + action_space_shape, device=device)
    logprobs = torch.zeros((num_steps, num_envs), device=device)
    rewards = torch.zeros((num_steps, num_envs), device=device)
    dones = torch.zeros((num_steps, num_envs), device=device)
    values = torch.zeros((num_steps, num_envs), device=device)

    return observations, actions, logprobs, rewards, dones, values


def lr_scheduler(optimizer, num_iterations, annealing = False):
    # iter_id is 0-indexed
    if annealing:
        frac_lambda = lambda epoch: 1.0 - epoch / num_iterations
    else:
        frac_lambda = lambda epoch: 1.0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = [frac_lambda])
