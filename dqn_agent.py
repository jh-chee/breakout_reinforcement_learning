import torch.optim as optim
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import random
from dqn import DQN
from replay_buffer import ReplayBuffer
from collections import deque


class DQNAgent:
    def __init__(self, action_size, history_size):
        self.render = False
        self.load_model = False

        self.action_size = action_size
        self.history_size = history_size

        self.discount_factor = 0.99
        self.learning_rate = 0.0001
        self.memory_size = 100000
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.explore_step = 1000000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.batch_size = 32
        self.train_start = 100000
        self.update_target = 1000

        self.memory = deque(maxlen=self.memory_size)
        self.replay_buffer = ReplayBuffer()

        # create main model and target model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DQN(action_size).to(self.device)
        self.target_model = DQN(action_size).to(self.device)

        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate)
        self.loss = nn.MSELoss()

        self.update_target_model()

        if self.load_model:
            self.model = torch.load('save_model/breakout_dqn')

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            state = torch.from_numpy(state).unsqueeze(0)
            state = Variable(state).float().to(self.device)
            action = self.model(state).data.cpu().max(1)[1]
            return int(action)

    def append_sample(self, history, action, reward, done):
        self.memory.append((history, action, reward, done))

    def get_sample(self, frame):
        mini_batch = []
        if frame >= self.memory_size:
            sample_range = self.memory_size
        else:
            sample_range = frame

        sample_range -= (self.history_size + 1)

        idx_sample = random.sample(range(sample_range), self.batch_size)
        for i in idx_sample:
            sample = []
            for j in range(self.history_size + 1):
                sample.append(self.memory[i + j])

            sample = np.array(sample)
            mini_batch.append((np.stack(sample[:, 0], axis=0), sample[3, 1], sample[3, 2], sample[3, 3]))

        return mini_batch

    def train_model(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = self.get_sample(frame)
        mini_batch = np.array(mini_batch).transpose()

        history = np.stack(mini_batch[0], axis=0)
        states = np.float32(history[:, :4, :, :]) / 255.
        actions = list(mini_batch[1])
        rewards = list(mini_batch[2])
        next_states = np.float32(history[:, 1:, :, :]) / 255.
        dones = mini_batch[3].astype(int)

        states = torch.Tensor(states)
        states = Variable(states).float().to(self.device)
        pred = self.model(states)

        a = torch.LongTensor(actions).view(-1, 1)
        one_hot_action = torch.FloatTensor(self.batch_size, self.action_size).zero_()
        one_hot_action.scatter_(1, a, 1)

        pred = torch.sum(pred.mul(Variable(one_hot_action).to(self.device)), dim=1)

        next_states = torch.Tensor(next_states)
        next_states = Variable(next_states).float().to(self.device)
        next_pred = self.target_model(next_states).data.cpu()

        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        target = rewards + (1 - dones) * self.discount_factor * next_pred.max(1)[0]
        target = Variable(target).to(self.device)

        self.optimizer.zero_grad()

        loss = self.loss(pred, target)
        loss.backward()

        # and train
        self.optimizer.step()
