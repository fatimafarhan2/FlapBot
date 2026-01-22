from collections import deque
import random
from turtle import done

class ReplayMemory:
    def __init__(self, maxlen, seed=None):

        self.memory = deque([],maxlen=maxlen)

        if seed is not None:
            random.seed(seed)

    def append(self,transition):
        """Save a transition"""
        self.memory.append(transition)

    def sample(self, sample_size):
        """Sample a batch of transitions"""
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)