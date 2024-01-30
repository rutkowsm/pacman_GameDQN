import math

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
from itertools import count
import cv2
import numpy as np

'''
pip install gymnasium
pip install gymnasium[classic-control]
pip install gymnasium[atari]
pip install gymnasium[accept-rom-license]
pip install torch
pip install opencv-python
'''

# Create the Pac-Man environment
env = gym.make("ALE/Pacman-v5", render_mode='human')


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(self._feature_size(input_shape), 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def _feature_size(self, input_shape):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *input_shape)))).view(1, -1).size(1)


def select_action(state, epsilon, dqn, num_actions):
    if random.random() > epsilon:
        with torch.no_grad():
            return dqn(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(num_actions)]], dtype=torch.long)


def compute_loss(batch, dqn, target_dqn, gamma):
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.cat(states)
    actions = torch.cat(actions)
    rewards = torch.cat(rewards)
    next_states = torch.cat(next_states)
    dones = torch.cat(dones)

    current_q_values = dqn(states).gather(1, actions)
    max_next_q_values = target_dqn(next_states).max(1)[0]
    expected_q_values = rewards + (gamma * max_next_q_values * (1 - dones))

    loss = F.mse_loss(current_q_values, expected_q_values.unsqueeze(1))
    return loss


def preprocess(frame, new_width=84, new_height=84):
    # Check if the frame is a NumPy array
    if not isinstance(frame, np.ndarray):
        raise ValueError("The frame is not in the expected format (NumPy array).")

    # Resize and convert to grayscale
    frame = cv2.cvtColor(cv2.resize(frame, (new_width, new_height)), cv2.COLOR_RGB2GRAY)

    # Normalize pixel values
    processed_frame = frame / 255.0

    return processed_frame


# Initialize the DQN
input_shape = (4, 84, 84)  # Adjust based on preprocessing
num_actions = env.action_space.n
dqn = DQN(input_shape, num_actions)
target_dqn = DQN(input_shape, num_actions)
target_dqn.load_state_dict(dqn.state_dict())
optimizer = torch.optim.Adam(dqn.parameters())
replay_buffer = ReplayBuffer(capacity=10000)

num_episodes = 1000  # Define the number of episodes for training
batch_size = 128  # Define the batch size for training
gamma = 0.99  # Discount factor for future rewards

for episode in range(num_episodes):
    state = env.reset()
    if isinstance(state, np.ndarray):
        state = preprocess(state)  # Preprocess if state is an image
    else:
        for t in count():
            # Example strategy
            epsilon_start = 1.0
            epsilon_final = 0.01
            epsilon_decay = 500
            epsilon = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * episode / epsilon_decay)

            action = select_action(state, epsilon, dqn, num_actions)
            next_state, reward, done, _ = env.step(action.item())[:4]
            next_state = preprocess(next_state)  # Preprocess next state

            replay_buffer.push(state, action, reward, next_state, done)

            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                loss = compute_loss(batch, dqn, target_dqn, gamma)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

            state = next_state
env.close()
