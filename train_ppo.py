import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import gym_carla
def main():
    env = gym.make("carla-v0")
    env = Monitor(env)
    env.reset()

    model = PPO(
        policy="MlpPolicy", env=env, learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
        vf_coef=0.5, max_grad_norm=0.5, verbose=1, tensorboard_log="./runs/ppo_BAYS3/", device="cuda",
    )
    model.learn(total_timesteps=100_000)
    model.save("ppo_bayes3_standard")


if __name__ == "__main__":
    main()
