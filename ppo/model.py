import math
import torch
import torch.nn as nn
from gymnasium.spaces.utils import flatdim
from ppo.utils import init_tensors
from torch.distributions.categorical import Categorical


@torch.no_grad()
def collect_policy_rollout(agent, env, obersavation, done, num_steps):
    num_envs = env.num_envs
    obs_space_shape = env.single_observation_space.shape
    action_space_shape = env.single_action_space.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (obs, acts, logprobs, rewards,
     dones, values) = init_tensors(num_steps, num_envs, obs_space_shape,
                                   action_space_shape, device)
    for step in range(num_steps):
        # initialization
        obs[step] = obersavation
        dones[step] = done
        # take an action
        action, logprob, entropy = agent.get_action(obersavation)
        value = agent.get_value(obersavation)
        acts[step] = action
        logprobs[step] = logprob
        # step the enviroment
        obersavation, reward, terminated, truncated, info = env.step(action.cpu().numpy())
        rewards[step] = torch.tensor(reward, device=device)
        obersavation = torch.tensor(obersavation, device=device)
        done = torch.tensor(terminated | truncated, device=device)

    return obs, acts, logprobs, rewards, dones, values, obersavation, done


@torch.no_grad()
def compute_advantage(rewards, num_steps, dones, values, next_done,
                      next_value, gamma = 0.99, gae = False):
    if not gae:
        returns = torch.zeros_like(rewards)
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - next_done.float()
                next_return = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                next_return = returns[t + 1]
            returns[t] = rewards[t] + gamma * nextnonterminal * next_return
        advantages = returns - values
    else:
        #TODO: implement GAE
        pass
    return advantages, returns


class Agent(nn.Module):
    def __init__(self, observation_space_n, action_space_n):
        super().__init__()
        # value function
        self.critic = nn.Sequential(
            nn.Linear(observation_space_n, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # policy function
        self.actor = nn.Sequential(
            nn.Linear(observation_space_n, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_space_n),
        )

        self.reset_parameters()

    def reset_parameters(self, std = math.sqrt(2), bias = 0.):
        for name, param in self.critic.named_parameters():
            if "weight" in name:
                idx = name.split(".")[0]
                std = std if idx != 2 else 1.0
                nn.init.orthogonal_(param, std)
            if "bias" in name:
                torch.nn.init.constant_(param, bias)

        for name, param in self.actor.named_parameters():
            if "weight" in name:
                idx = name.split(".")[0]
                std = std if idx != 2 else 0.01
                nn.init.orthogonal_(param, std)
            if "bias" in name:
                torch.nn.init.constant_(param, bias)


    def get_value(self, obersavation):
        return self.critic(obersavation).squeeze(dim=-1)

    def get_action(self, obersavation, action=None):
        logits = self.actor(obersavation)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy()
