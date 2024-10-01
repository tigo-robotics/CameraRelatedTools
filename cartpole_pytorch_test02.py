import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def run(is_training=True, render=False):
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    if is_training:
        model = QNetwork(state_size, action_size)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
    else:
        with open('cartpole_model.pth', 'rb') as f:
            model = QNetwork(state_size, action_size)
            model.load_state_dict(torch.load(f))
            model.eval()

    learning_rate_a = 0.1
    discount_factor_g = 0.99
    epsilon = 1.0
    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()
    rewards_per_episode = []

    i = 0

    while True:
        state = env.reset()[0]
        rewards = 0
        terminated = False

        while not terminated and rewards < 100:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = torch.argmax(model(state_tensor)).item()

            new_state, reward, terminated, _, _ = env.step(action)
            rewards += reward
            new_state_tensor = torch.FloatTensor(new_state).unsqueeze(0)

            if is_training:
                # Update Q-values
                target = reward + discount_factor_g * torch.max(model(new_state_tensor)).item()
                output = model(state_tensor)[0][action]
                loss = criterion(output, torch.tensor(target))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = new_state

        rewards_per_episode.append(rewards)
        mean_rewards = np.mean(rewards_per_episode[-100:])

        if is_training and i % 100 == 0:
            print(f'Episode: {i} {rewards}  Epsilon: {epsilon:.2f}  Mean Rewards {mean_rewards:.1f}')

        if mean_rewards > 195:  # CartPole-v1 benchmark
            break

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        i += 1

    env.close()

    if is_training:
        torch.save(model.state_dict(), 'cartpole_model.pth')

    mean_rewards = []
    for t in range(i):
        mean_rewards.append(np.mean(rewards_per_episode[max(0, t - 100):(t + 1)]))
    plt.plot(mean_rewards)
    plt.savefig('cartpole.png')

if __name__ == '__main__':
    run(is_training=True, render=1)
    # For inference
    #run(is_training=False, render=True)
