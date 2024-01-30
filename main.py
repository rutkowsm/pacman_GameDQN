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
    # If frame is not a NumPy array, attempt conversion
    if not isinstance(frame, np.ndarray):
        frame = np.array(frame)

    # Check if the frame is grayscale (2D) or color (3D)
    if len(frame.shape) == 3 and frame.shape[2] == 3:  # If color image
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    elif len(frame.shape) == 2 or frame.shape[2] == 1:  # If already grayscale
        pass  # No color conversion needed
    else:
        raise ValueError("Unexpected frame format. Check the environment's output.")

    # Resize the frame
    frame = cv2.resize(frame, (new_width, new_height))

    # Normalize pixel values
    processed_frame = frame / 255.0

    return processed_frame

def preprocess_frame_in_tuple(state_tuple, new_width=84, new_height=84):
    frame = state_tuple[0]  # Extract the frame from the tuple

    # Check if the frame is already grayscale (single channel)
    if frame.ndim == 3 and frame.shape[2] == 3:  # If it's a color image (RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Resize the frame
    frame = cv2.resize(frame, (new_width, new_height))

    # Normalize pixel values
    processed_frame = frame / 255.0

    return processed_frame

# Initialize the DQN
height = 84
width = 84
input_shape = (4, height, width)  # Adjust based on preprocessing
num_actions = env.action_space.n
dqn = DQN(input_shape, num_actions)
target_dqn = DQN(input_shape, num_actions)
target_dqn.load_state_dict(dqn.state_dict())
optimizer = torch.optim.Adam(dqn.parameters())
replay_buffer = ReplayBuffer(capacity=10000)

# Epsilon decay parameters
epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500

num_stacked_frames = 4
frame_stack = deque(maxlen=num_stacked_frames)
num_episodes = 1000
batch_size = 128  # Define the batch size for training
gamma = 0.99  # Discount factor for future rewards

for episode in range(num_episodes):
    epsilon = epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * episode / epsilon_decay)
    state_tuple = env.reset()
    state = preprocess_frame_in_tuple(state_tuple)
    frame_stack.clear()
    for _ in range(num_stacked_frames):
        frame_stack.append(state)

    for t in count():
        if len(frame_stack) == num_stacked_frames:
            state_array = np.array(list(frame_stack)).reshape(1, num_stacked_frames, height, width)
            state_tensor = torch.tensor(state_array, dtype=torch.float)

            action = select_action(state_tensor, epsilon, dqn, num_actions)
            next_state_tuple, reward, done, _ = env.step(action.item())[:4]  # Step returns a tuple
            next_state = preprocess_frame_in_tuple(next_state_tuple)
            frame_stack.append(next_state)

            next_state_array = np.array(list(frame_stack)).reshape(1, num_stacked_frames, height, width)
            next_state_tensor = torch.tensor(next_state_array, dtype=torch.float)

            action_tensor = torch.tensor([action.item()], dtype=torch.long)
            reward_tensor = torch.tensor([reward], dtype=torch.float)
            done_tensor = torch.tensor([done], dtype=torch.float)

            replay_buffer.push(state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor)

            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                loss = compute_loss(batch, dqn, target_dqn, gamma)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        state = next_state  # Correct indentation

env.close()
