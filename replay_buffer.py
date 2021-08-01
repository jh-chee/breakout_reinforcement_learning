from collections import deque
import numpy as np
import random


class ReplayBuffer:
    def __init__(self, max_len=1000000):
        self.buffer = deque(maxlen=max_len)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        sample_size = min(len(self.buffer), batch_size)
        sample_indices = random.choices(range(len(self.buffer)), k=sample_size)
        samples = np.array(self.buffer, dtype=object)[sample_indices]
        return list(zip(*samples))

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, max_len):
        self.buffer = deque(maxlen=max_len)
        self.priorities = deque(maxlen=max_len)

    def add(self, experience): # experience = (state, action, next_state, reward, done)
        self.buffer.append(experience)
        self.priorities.append(max(self.priorities, default=1))

    def get_probabilities(self, priority_scale):
        scaled_priorities = np.array(self.priorities) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities

    def get_importance(self, probabilities):
        importance = 1 / len(self.buffer) * 1 / probabilities
        importance_normalized = importance / max(importance)
        return importance_normalized

    # returns (states, actions, next_states, rewards, dones), importance, indices
    def sample(self, batch_size, priority_scale=1.0):
        sample_size = min(len(self.buffer), batch_size)
        sample_probs = self.get_probabilities(priority_scale)
        sample_indices = random.choices(range(len(self.buffer)), k=sample_size, weights=sample_probs)
        samples = np.array(self.buffer)[sample_indices]
        importance = self.get_importance(sample_probs[sample_indices])
        return map(list, zip(*samples)), importance, sample_indices

    def set_priorities(self, indices, errors, offset=0.1):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset
