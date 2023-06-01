import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym


def make_vec_env(env_id: str, num_env: int, seed: int = 1024, start_index: int = 0,
                 output_dir: str = "./", capture_video: bool = False):
    def _make_env(rank, env_idx):
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.record_episode_statistics.RecordEpisodeStatistics(env)
        if capture_video and num_env == 1:
            env = gym.wrappers.RecordVideo(env, f"{output_dir}/videos/{env_id}")
        # seed everything
        env.reset(seed = rank + seed)
        env.action_space.seed(rank + seed)
        env.observation_space.seed(rank + seed)
        return env

    if num_env == 1:
        return _make_env(start_index, 0)

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


@torch.no_grad()
def evaluate_model(agent, env, n_eval_episodes, return_episode_rewards = False):
    episode_rewards = []
    episode_lengths = []
    cur_reward, cur_length, episode_counter = 0, 0, 0
    observation, _ = env.reset()
    terminated, truncated = False, False
    device = next(agent.parameters()).device

    while episode_counter < n_eval_episodes:
        while not (terminated | truncated):
            observation = torch.tensor(observation, device=device)
            action, logprob, entropy = agent.get_action(observation)
            observation, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            cur_reward += reward
            cur_length += 1
            # one episode ends
            if terminated | truncated:
                episode_rewards.append(cur_reward)
                episode_lengths.append(cur_length)

                cur_reward = 0
                cur_length = 0
                episode_counter += 1

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    if return_episode_rewards:
        return episode_rewards, episode_lengths

    return mean_reward, std_reward
