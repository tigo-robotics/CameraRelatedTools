#pip install gymnasium torch matplotlib numpy
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import pickle

# Modify the Q-Network to handle one-hot encoding
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)  # Input size matches state size (64 for 8x8)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Training function
def train_agent(episodes, render=False):
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode='human' if render else None)
    env.metadata['render_fps']=300
    state_size = env.observation_space.n  # Should be 64 for 8x8
    action_size = env.action_space.n

    q_network = QNetwork(state_size, action_size)
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    replay_memory = deque(maxlen=2000)
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.01
    discount_factor = 0.99

    rewards_per_episode = []

    for episode in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    # Convert state to one-hot encoding
                    state_tensor = torch.zeros(state_size)
                    state_tensor[state] = 1.0
                    action = torch.argmax(q_network(state_tensor)).item()

            # Take action
            new_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

            # Store transition in replay memory
            replay_memory.append((state, action, reward, new_state, done))

            # Sample from replay memory
            if len(replay_memory) > 32:
                batch = random.sample(replay_memory, 32)
                for s, a, r, s_next, d in batch:
                    # Convert states to one-hot encoding
                    s_tensor = torch.zeros(state_size)
                    s_tensor[s] = 1.0
                    s_next_tensor = torch.zeros(state_size)
                    s_next_tensor[s_next] = 1.0

                    target = r + (1 - d) * discount_factor * torch.max(q_network(s_next_tensor)).item()
                    target_f = q_network(s_tensor)
                    target_f[a] = target

                    # Update the network
                    optimizer.zero_grad()
                    loss = criterion(q_network(s_tensor), target_f)
                    loss.backward()
                    optimizer.step()

            state = new_state

        # Decay epsilon
        if epsilon > min_epsilon:
            epsilon *= epsilon_decay

        rewards_per_episode.append(total_reward)
        print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}")

    env.close()

    # Plot rewards
    plt.plot(rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.show()

if __name__ == '__main__':
    train_agent(16000, render=1)