import hydra
import logging
import torch
import random
from configs.schema import TrainingConfig
from hydra.core.config_store import ConfigStore
from ppo.utils import make_vec_env, lr_scheduler, evaluate_model
from ppo.model import Agent, collect_policy_rollout, compute_advantage
from ppo.losses import CLIPLoss, VFLoss
from torch.utils.tensorboard import SummaryWriter


cs = ConfigStore.instance()
cs.store(name="config", node=TrainingConfig)

@hydra.main(version_base=None, config_name="config")
def train(cfg: TrainingConfig) -> None:
    writer = SummaryWriter(log_dir=cfg.output_dir)
    # step 0: get the device info
    if not torch.cuda.is_available and cfg.device == "cuda":
        logging.warning("cuda is unavailable!... switching to cpu instead.")
        cfg.device = "cpu"
    device = cfg.device

    # step 1: make environment and initialize observation & done
    logging.info("step 1: creating the environment")
    env = make_vec_env(cfg.env_id, cfg.num_envs, cfg.seed, capture_video=cfg.capture_video)
    init_observation, _ = env.reset()
    init_done = torch.zeros(cfg.num_envs, device=device)
    init_observation = torch.tensor(init_observation, device=device)

    # step 2: instantiate agent
    logging.info("step 2: instantiating the A2C")
    agent = Agent(env).to(device)

    # step 3: instantiate optimizer, loss and lr scheduler
    logging.info("step 3: instantiating optimizer, scheduler and loss")
    optimizer = torch.optim.Adam(agent.parameters(), cfg.learning_rate, eps=1e-5)
    scheduler = lr_scheduler(optimizer, cfg.num_iterations, annealing = cfg.anneal_lr)
    clip_loss = CLIPLoss().to(device)
    vf_loss = VFLoss().to(device)
    n_steps = 0
    avg_loss = 0

    # step 4: loop for training, including rollout data collection,
    # advantage computation and gradient descent
    logging.info("step 4: starting data collection and training")
    for it in range(cfg.num_iterations):
        (observations, actions, logprobs,
         rewards, dones, values, last_observation,
         last_done) = collect_policy_rollout(agent, env, init_observation,
                                             init_done, cfg.rollout_length)

        last_value = agent.get_value(last_observation)
        advantage, returns = compute_advantage(rewards, cfg.rollout_length, dones, values,
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
            value_loss = vf_loss(returns[indices], cur_values)
            policy_loss = clip_loss(normed_advantage, ratio, cfg.clip_epsilon)
            entropy_loss = entropy.mean()
            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy_loss

            n_steps += len(observations)
            avg_loss += loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), cfg.grad_norm)
            optimizer.step()

            # log as much info as possible
            writer.add_scalar("losses/value_loss", value_loss.item(), n_steps)
            writer.add_scalar("losses/policy_loss", policy_loss.item(), n_steps)
            writer.add_scalar("losses/entropy_loss", entropy_loss.item(), n_steps)
            writer.add_scalar("losses/total_loss", loss.item(), n_steps)

        if (it + 1) % cfg.logging_iterations == 0:
            avg_loss = avg_loss / (len(observations) * cfg.logging_iterations * cfg.epochs)
            logging.info(f"iteration: {it}, logging average loss: {avg_loss}")
            avg_loss = 0
        scheduler.step()
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], n_steps)

    writer.close()

    logging.info("preparing evaluation environment")
    eval_env = make_vec_env(cfg.env_id, num_env=1,
                            output_dir = cfg.output_dir, capture_video=True)

    eval_env.close()
    mean_reward, std_reward = evaluate_model(agent, eval_env,
                                             cfg.num_eval_episodes,
                                             )
    logging.info(f"evaluation mean reward {mean_reward}, std {std_reward}")
    logging.info("saving the model to the destination")
    torch.save(agent.state_dict(), f"{cfg.output_dir}/ppo.pt")

if __name__ == "__main__":
    train()
