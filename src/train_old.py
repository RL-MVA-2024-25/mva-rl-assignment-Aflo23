import matplotlib.pyplot as plt
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)


# Définition du Q-Network avec 2 couches LSTM
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.fc2 = nn.Linear(128, action_size)

    def forward(self, x, hidden1=None, hidden2=None):
        x = torch.relu(self.fc1(x))
        x = x.unsqueeze(1)  # Ajout de la dimension séquence pour le LSTM (seq_len=1)
        if hidden1 is None:
            x, hidden1 = self.lstm1(x)
        else:
            x, hidden1 = self.lstm1(x, hidden1)
        if hidden2 is None:
            x, hidden2 = self.lstm2(x)
        else:
            x, hidden2 = self.lstm2(x, hidden2)
        x = x.squeeze(1)  # Retirer la dimension séquence après le LSTM
        return self.fc2(x), hidden1, hidden2


# Buffer d'expérience pour le replay
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32),
        )

    def size(self):
        return len(self.buffer)


# Agent DQN
"""
epsilon decay updaté de 
epsilon start.

lr à 1e-4 comparé à avant 1e-3

"""
class ProjectAgent:
    def __init__(self, gamma=0.99, lr=1e-4, buffer_size=5000, batch_size=64, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.001):
        self.state_size = env.unwrapped.observation_space.shape[0]
        self.action_size = env.unwrapped.action_space.n
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.q_network = QNetwork(self.state_size, self.action_size)
        self.target_network = QNetwork(self.state_size, self.action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = ReplayBuffer(buffer_size, batch_size)

    def act(self, state, hidden1=None, hidden2=None):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1), hidden1, hidden2
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values, hidden1, hidden2 = self.q_network(state, hidden1, hidden2)
        return torch.argmax(q_values).item(), hidden1, hidden2

    def learn(self):
        if self.memory.size() < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample()
        q_values, _, _ = self.q_network(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values, _, _ = self.target_network(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        loss = self.criterion(q_values, target_q_values)

        # Logarithme de la perte et renormalisation
        log_loss = torch.log1p(loss)  # Logarithme pour stabiliser la perte
        normalized_loss = (log_loss - log_loss.min()) / (log_loss.max() - log_loss.min())  # Normalisation

        self.optimizer.zero_grad()
        normalized_loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())


# Boucle d'entraînement
def train_dqn(env=env, episodes=1000, update_freq=10):
    agent = ProjectAgent()
    rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False
        hidden1, hidden2 = None, None

        while not done and not truncated:
            action, hidden1, hidden2 = agent.act(state, hidden1, hidden2)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.memory.add(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            agent.learn()

        rewards.append(total_reward)
        agent.epsilon = max(agent.epsilon_min, agent.epsilon - agent.epsilon_decay)

        if episode % update_freq == 0:
            agent.update_target_network()

        print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

    return rewards


if __name__ == "__main__":
    env = TimeLimit(HIVPatient(domain_randomization=False), max_episode_steps=200)
    rewards = train_dqn(env=env, episodes=1000, update_freq=10)
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Training Rewards")
    plt.show()
