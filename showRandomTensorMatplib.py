import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pyautogui
import numpy as np
import cv2
import gym
from gym import spaces
import time

# Screen capture settings
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# Hyperparameters
LR = 1e-3
GAMMA = 0.99
EPSILON = 0.1
TARGET_UPDATE = 10

# Custom environment for mouse movement
class MouseEnv(gym.Env):
    def __init__(self):
        super(MouseEnv, self).__init__()
        # Observation space: screen pixels (downscaled to 80x80 grayscale)
        self.observation_space = spaces.Box(low=0, high=255, shape=(80, 80, 1), dtype=np.uint8)
        # Action space: move the mouse up/down/left/right
        self.action_space = spaces.Discrete(4)
        self.target_pos = np.random.randint(0, SCREEN_WIDTH), np.random.randint(0, SCREEN_HEIGHT)
    
    def step(self, action):
        # Take action: move mouse
        x, y = pyautogui.position()
        if action == 0:
            x += 10  # move right
        elif action == 1:
            x -= 10  # move left
        elif action == 2:
            y += 10  # move down
        elif action == 3:
            y -= 10  # move up
        pyautogui.moveTo(x, y)

        # Get observation (screen capture)
        observation = self._get_observation()

        # Calculate reward (distance to target)
        current_pos = np.array([x, y])
        target_pos = np.array(self.target_pos)
        distance = np.linalg.norm(current_pos - target_pos)
        reward = -distance

        # Check if done (if mouse is close to target)
        done = distance < 20
        if done:
            reward += 100  # bonus for reaching the target
            self.target_pos = np.random.randint(0, SCREEN_WIDTH), np.random.randint(0, SCREEN_HEIGHT)

        return observation, reward, done, {}

    def reset(self):
        self.target_pos = np.random.randint(0, SCREEN_WIDTH), np.random.randint(0, SCREEN_HEIGHT)
        return self._get_observation()

    def _get_observation(self):
        # Capture the screen and resize it to 80x80 grayscale
        screen = np.array(pyautogui.screenshot())
        screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen_resized = cv2.resize(screen_gray, (80, 80))
        return np.expand_dims(screen_resized, axis=-1)

# Simple neural network for RL agent
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(32 * 17 * 17, 256)
        self.fc2 = nn.Linear(256, 4)  # 4 actions

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Replay buffer for experience replay
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return zip(*random.sample(self.memory, batch_size))

    def __len__(self):
        return len(self.memory)

# Select an action based on epsilon-greedy strategy
def select_action(state, policy_net, steps_done, n_actions):
    global EPSILON
    sample = np.random.rand()
    epsilon_threshold = EPSILON * (1.0 / (1.0 + steps_done))
    if sample > epsilon_threshold:
        with torch.no_grad():
            return policy_net(state).argmax(dim=1).view(1, 1)
    else:
        return torch.tensor([[np.random.randint(0, n_actions)]], dtype=torch.long)

# Training function for DQN
def optimize_model(memory, policy_net, target_net, optimizer, batch_size):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = transitions

    state_batch = torch.cat(state_batch)
    next_state_batch = torch.cat(next_state_batch)
    action_batch = torch.cat(action_batch)
    reward_batch = torch.cat(reward_batch)
    done_batch = torch.cat(done_batch)

    # Compute Q(s_t, a)
    q_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states
    next_state_values = torch.zeros(batch_size)
    next_state_values[~done_batch] = target_net(next_state_batch[~done_batch]).max(1)[0]

    # Compute expected Q values
    expected_q_values = reward_batch + (GAMMA * next_state_values)

    # Loss function
    loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Main RL loop
def main():
    env = MouseEnv()
    n_actions = env.action_space.n

    policy_net = DQN().cuda()
    target_net = DQN().cuda()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(10000)

    steps_done = 0
    for episode in range(1000):
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).cuda()

        for t in range(100):
            # Select and perform an action
            action = select_action(state, policy_net, steps_done, n_actions)
            next_state, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], dtype=torch.float32).cuda()

            # Observe new state
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).cuda()
            done_tensor = torch.tensor([done], dtype=torch.bool).cuda()

            # Store the transition in memory
            memory.push(state, action, reward, next_state, done_tensor)

            # Move to the next state
            state = next_state

            # Perform one step of optimization
            optimize_model(memory, policy_net, target_net, optimizer, batch_size=32)

            if done:
                break

        # Update target network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

if __name__ == "__main__":
    main()
import torch
import matplotlib.pyplot as plt

def generate_random_tensor(rows, cols):
    return torch.rand(rows, cols)

def display_tensor(tensor):
    plt.imshow(tensor, cmap='viridis')
    plt.colorbar()
    plt.title('Random Tensor Visualization')
    plt.show()

def main():
    rows, cols = 10, 10  # Increased size for better visualization
    tensor = generate_random_tensor(rows, cols)
    
    print("Random Tensor:")
    print(tensor)
    
    display_tensor(tensor)

if __name__ == "__main__":
    main()import torch
import matplotlib.pyplot as plt

def generate_random_tensor(rows, cols):
    return torch.rand(rows, cols)

def display_tensor(tensor):
    plt.imshow(tensor, cmap='viridis')
    plt.colorbar()
    plt.title('Random Tensor Visualization')
    plt.show()

def main():
    rows, cols = 10, 10  # Increased size for better visualization
    tensor = generate_random_tensor(rows, cols)
    
    print("Random Tensor:")
    print(tensor)
    
    display_tensor(tensor)

if __name__ == "__main__":
    main()import torch
import matplotlib.pyplot as plt

def generate_random_tensor(rows, cols):
    return torch.rand(rows, cols)

def display_tensor(tensor):
    plt.imshow(tensor, cmap='viridis')
    plt.colorbar()
    plt.title('Random Tensor Visualization')
    plt.show()

def main():
    rows, cols = 10, 10  # Increased size for better visualization
    tensor = generate_random_tensor(rows, cols)
    
    print("Random Tensor:")
    print(tensor)
    
    display_tensor(tensor)

if __name__ == "__main__":
    main()