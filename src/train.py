import pickle
import numpy as np
#from fast_env import HIVPatient
from gymnasium.wrappers import TimeLimit
from fast_env import FastHIVPatient
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO

env = TimeLimit(
    env=FastHIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.

# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:

    def __init__(self):
        self.env = TimeLimit(FastHIVPatient(), max_episode_steps=200)
        self.model = None
        self.obs_mean = None
        self.obs_std = None
        self.obs_var = None

    def apply_obs_normalization(self, obs):
        return (obs - self.obs_mean) / self.obs_std

    def act(self, observation, use_random=False):
        if use_random:
            return self.env.action_space.sample()
        else:
            observation = self.apply_obs_normalization(observation)
            action, _state = self.model.predict(observation, deterministic=True)
            return action

    def save(self, path):
        self.model.save(path + 'ppo_model')
        self.env.save(path + "vec_normalize_stats.pkl")

    def load(self):
        self.model = PPO.load("ppo_model")
        vec_norm = pickle.load(open("vec_normalize_stats.pkl", "rb"))

        self.obs_mean = vec_norm.obs_rms.mean
        self.obs_var = vec_norm.obs_rms.var
        self.obs_std = np.sqrt(self.obs_var + 1e-8)

    def linear_schedule(self, initial_lr):
        def scheduler(progress_remaining):
            return progress_remaining * initial_lr
        return scheduler

    def exponential_schedule(self, initial_lr, decay_rate):
        def scheduler(progress_remaining):
            return initial_lr * (decay_rate ** (1 - progress_remaining))
        return scheduler

    def make_hiv_env(self, seed: int):
        def _init():
            env = FastHIVPatient(domain_randomization=True)
            env = TimeLimit(env, max_episode_steps=200)
            env.reset(seed=seed)
            return env
        return _init

    def training(self, iterations=5_000_000, gamma=0.99, disable_tqdm=False):
        num_envs = 8
        envs = [self.make_hiv_env(i) for i in range(num_envs)]
        vec_env = DummyVecEnv(envs)
        self.env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        eval_cb = EvalCallback(
            self.env,
            best_model_save_path="./logs/",
            log_path="./logs/",
            eval_freq=10_000,
            n_eval_episodes=10,
            deterministic=True
        )

        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        )
        '''
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            n_steps=4096,
            batch_size=256,
            learning_rate=self.linear_schedule(3e-4),
            gamma=gamma,
            gae_lambda=0.95,
            clip_range=0.25,
            ent_coef=0.1,
            n_epochs=20,
            verbose=not disable_tqdm,
            policy_kwargs=policy_kwargs,
            vf_coef = 1.0,
            tensorboard_log="./ppo_hiv_tensorboard/"
        )
        '''
        self.model.env = self.env
        self.model.learn(total_timesteps=iterations, callback=eval_cb)

# agent = ProjectAgent()
# agent.load()
# #agent.collect_samples(10000)
# agent.training()
# agent.save('')
# agent.load()
# #agent.model = PPO.load('logs/best_model.zip')


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


