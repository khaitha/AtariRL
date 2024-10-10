import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple,deque
from itertools import count
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

env = gym.make("ALE/Breakout-v5")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations,n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 128)
        self.layer4 = nn.Linear(128, n_actions)

    def forward(self,x):
        x = F.leaky_relu(self.layer1(x))
        x = F.leaky_relu(self.layer2(x))
        x = F.leaky_relu(self.layer3(x))
        return self.layer4(x)

batch_size = 63
gamma = 0.99
eps_start = .9
eps_end = .05
eps_decay = 1000
tau = .005
lr = .0001

n_actions = env.action_space.n

state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations,n_actions).to(device)
target_net = DQN(n_observations,n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = eps_start + (eps_end - eps_start) * math.exp(-1. * steps_done / eps_decay)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            action = policy_net(state).argmax(dim=1, keepdim=True)
            return action
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

    episode_durations = []

    def plot_durations(show_result = False):
        plt.figure(1)
        duration_t = torch.tensor(episode_durations, dtype = torch.float)
        if show_result:
            plt.title("Result")
        else:
            plt.clf()
            plt.title("Training...")
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(duration_t.numpy())

        if len(duration_t) >= 100:
            means = duration_t.unfold(0, 100, 1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(.001)
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)

            else:
                display.display(plt.gcf())

def optimize_model():
    if len(memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device = device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Q(s_t,a)
    state_action_value = policy_net(state_batch).gather(1, action_batch)

    # V(s_{t+1})
    next_state_value = torch.zeros(batch_size, n_actions).to(device)

    with torch.no_grad():
        next_state_action_value = target_net(non_final_mask).values

    expected_state_action_value = (next_state_value * gamma) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_value, expected_state_action_value.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 100)
    optimizer.step()

# ------- Run ------------- #
num_episodes = 50

for i in range(num_episodes):

    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    for c in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _= env.step(action)
        reward = torch.tensor([reward], device = device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tesnor(observation, dtype = torch.float32, device = device).unsqueeze(0)
