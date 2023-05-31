import hydra
import torch
import random
from configs.schema import TrainingConfig
from hydra.core.config_store import ConfigStore
from ppo.utils import make_vec_env, lr_scheduler
from ppo.model import Agent, collect_policy_rollout, compute_advantage
from ppo.losses import CLIPLoss, VFLoss
from torch.utils.tensorboard import SummaryWriter


cs = ConfigStore.instance()
cs.store(name="config", node=TrainingConfig)

@hydra.main(version_base=None, config_name="config")
def train(cfg: TrainingConfig) -> None:
    writer = SummaryWriter(log_dir=cfg.output_dir)

    # step 1: make environment and initialize observation & done
    env = make_vec_env(cfg.env_id, cfg.num_envs, cfg.seed, capture_video=cfg.capture_video)
    init_observation, _ = env.reset()
    init_done, init_observation = torch.zeros(cfg.num_envs), torch.tensor(init_observation)
    # step 2: instantiate agent
    agent = Agent(env)
    # step 3: instantiate optimizer, loss and lr scheduler
    optimizer = torch.optim.Adam(agent.parameters(), cfg.learning_rate, eps=1e-5)
    scheduler = lr_scheduler(optimizer, cfg.num_iterations, annealing = cfg.anneal_lr)
    clip_loss = CLIPLoss()
    vf_loss = VFLoss()
    # step 4: loop for training, including rollout data collection,
    # advantage computation and gradient descent
    for it in range(cfg.num_iterations):
        (observations, actions, logprobs,
         rewards, dones, values, last_observation,
         last_done) = collect_policy_rollout(agent, env, init_observation,
                                             init_done, cfg.num_steps)

        last_value = agent.get_value(last_observation)
        advantage, returns = compute_advantage(rewards, cfg.num_steps, dones, values,
                                               last_done, last_value, gamma=cfg.gamma)

        # reshape the tensors
        observations = observations.reshape(-1, *env.single_observation_space.shape)
        actions = actions.reshape((-1, *env.single_action_space.shape))
        logprobs = logprobs.flatten()
        rewards = rewards.flatten()
        values = values.flatten()
        returns = returns.flatten()
        advantage = advantage.flatten()

        for epoch in range(cfg.epochs):
            indices = list(range(observations.shape[0]))
            random.shuffle(indices)
            cur_values = agent.get_value(observations[indices])
            _, cur_logprobs, entropy = agent.get_action(observations[indices],
                                                        actions[indices])
            ratio = cur_logprobs - logprobs
            # TODO: KL regularization

            # normalize advantage by default
            shuffled_advantage = advantage[indices]
            mean, std = shuffled_advantage.mean(), shuffled_advantage.std()
            normed_advantage = (shuffled_advantage - mean) / (std + 1e-8)

            # loss term
            loss = (clip_loss(normed_advantage, ratio, cfg.clip_epsilon) +
                    cfg.vf_coef * vf_loss(returns[indices], cur_values) -
                    cfg.ent_coef * entropy.mean())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.grad_norm)
            optimizer.step()

        scheduler.step()

    writer.close()


if __name__ == "__main__":
    train()
