from sched import scheduler

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.callbacks import EvalCallback
from torch.backends.cudnn import deterministic

from env_hiv import HIVPatient
from fast_env_2 import FastHIVPatient
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import random
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import SAC
from stable_baselines3 import PPO

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.



# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:

    def __init__(self):
        self.capacity = 10000
        self.capQ = 10
        self.index = 0
        self.idQ = 0
        self.S = []
        self.A = []
        self.R = []
        self.S2 = []
        self.D = []
        self.Qfunctions = []
        self.env = TimeLimit(FastHIVPatient(), max_episode_steps=200)
        self.model = None

    def act(self, observation, use_random=False):
        if use_random:
            return self.env.action_space.sample()
        else:
            action, _state = self.model.predict(observation, deterministic=True)
            return action

    def save(self, path):
        self.model.save('ppo_model')
        self.env.save("vec_normalize_stats.pkl")

    def load(self):
        self.model = PPO.load("ppo_model")

    def collect_samples(self, horizon, disable_tqdm=False, print_done_states=False):
        s, _ = self.env.reset()
        # dataset = []
        self.S = []
        self.A = []
        self.R = []
        self.S2 = []
        self.D = []
        for _ in tqdm(range(horizon), disable=disable_tqdm):
            a = self.env.action_space.sample()
            s2, r, done, trunc, _ = self.env.step(a)
            # dataset.append((s,a,r,s2,done,trunc))

            if len(self.S) < self.capacity:
                self.S.append(None)
                self.A.append(None)
                self.R.append(None)
                self.S2.append(None)
                self.D.append(None)
            self.S[self.index] = s
            self.A[self.index] = a
            self.R[self.index] = r
            self.S2[self.index] = s2
            self.D[self.index] = done
            self.index = (self.index + 1) % self.capacity

            if done or trunc:
                s, _ = self.env.reset()
                if done and print_done_states:
                    print("done!")
            else:
                s = s2
        self.S = np.array(self.S)
        self.A = np.array(self.A).reshape((-1, 1))
        self.R = np.array(self.R)
        self.S2 = np.array(self.S2)
        self.D = np.array(self.D)

    def make_env(self):
        env = TimeLimit(FastHIVPatient(domain_randomization=True), max_episode_steps=200)
        return env

    def linear_schedule(self, initial_lr):
        def scheduler(progress_remaining):
            return progress_remaining * initial_lr

        return scheduler

    def exponential_schedule(self, initial_lr, decay_rate):
        def scheduler(progress_remaining):
            return initial_lr * (decay_rate ** (1 - progress_remaining))

        return scheduler

    def linear_entropy_schedule(self, start=0.20, end=0.01):
        """
        Returns a function that computes a linearly decreasing entropy coefficient
        from 'start' to 'end' over the course of training.
        """

        def schedule(progress_remaining: float) -> float:
            # progress_remaining = 1.0 at start, 0.0 at end
            return end + (start - end) * progress_remaining

        return schedule


    def training(self, iterations, gamma, disable_tqdm=False):
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        )
        orig_env = TimeLimit(FastHIVPatient(), max_episode_steps=200)
        vec_env = DummyVecEnv([lambda: orig_env])
        self.env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        eval_cb = EvalCallback(
            self.env,
            best_model_save_path="./logs/",
            log_path="./logs/",
            eval_freq=10_000,
            n_eval_episodes=10,
            deterministic=True
        )
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            n_steps=4096,           # how many steps to collect per update
            batch_size=256,          # mini-batch size
            learning_rate=self.linear_schedule(3e-4),     # try 1e-4 or 3e-4
            gamma=0.99,             # discount factor
            gae_lambda=0.95,        # GAE parameter
            clip_range=0.4,         # clipping parameter
            ent_coef=0.2,          # encourage some exploration
            n_epochs=20,            # number of epochs per update
            verbose=1,
            policy_kwargs=policy_kwargs,
            vf_coef = 1.0,
            tensorboard_log="./ppo_hiv_tensorboard/"
        )
        self.model.learn(total_timesteps=5_000_000, callback=eval_cb)

# print('First step')
# agent = ProjectAgent()
# #agent.collect_samples(10000)
# agent.training(1700, 0.9)
# agent.save('/Users/antoine/Downloads/RL_project/mva-rl-assignment-Aflo23/src/')
# agent.load()
# #agent.model = PPO.load('logs/best_model.zip')

# print('Second step: preparing environment/ EnV determined')
# eval_env = DummyVecEnv([lambda: TimeLimit(FastHIVPatient(), max_episode_steps=200)])
# eval_env = VecNormalize.load("vec_normalize_stats.pkl", eval_env)
# eval_env.training = False   # turn off updates to running stats
# eval_env.norm_reward = False # or True, depending on how you want to log
# rewards: list[float] = []
# for _ in range(5):
#     obs = eval_env.reset()
#     done = False
#     truncated = False
#     episode_reward = 0
#     while not done and not truncated:
#         action = agent.act(obs)
#         print(action)
#         obs, reward, done,  _ = eval_env.step(action)
#         episode_reward += reward
#     rewards.append(episode_reward)
# print(np.mean(rewards))

# print('Third step: preparing environment/Env random')
# eval_env = DummyVecEnv([lambda: TimeLimit(FastHIVPatient(domain_randomization=True), max_episode_steps=200)])
# eval_env = VecNormalize.load("vec_normalize_stats.pkl", eval_env)
# eval_env.training = False   # turn off updates to running stats
# eval_env.norm_reward = False # or True, depending on how you want to log
# rewards: list[float] = []
# for _ in range(20):
#     obs = eval_env.reset()
#     done = False
#     truncated = False
#     episode_reward = 0
#     while not done and not truncated:
#         action = agent.act(obs)
#         obs, reward, done,  _ = eval_env.step(action)
#         episode_reward += reward
#     rewards.append(episode_reward)
# print(np.mean(rewards))


