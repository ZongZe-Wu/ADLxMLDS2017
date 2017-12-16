import os
import random
import logging
import numpy as np


class ReplayMemory:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.states = np.empty((self.memory_size, 84, 84, 4), dtype = np.float32)
        self.actions = np.empty(self.memory_size, dtype = np.int32)
        self.rewards = np.empty(self.memory_size, dtype = np.float32)
        self.terminals = np.empty(self.memory_size, dtype = np.bool)
        self.current = 0

        #self.pre_states = np.empty((self.batch_size, 84, 84, 4), dtype = np.float32)
        #self.post_states = np.empty((self.batch_size, 84, 84, 4), dtype = np.float32)
    def store(self, s, a, r, t):
        index = (self.current) % self.memory_size
        self.states[index, ...] = s
        self.actions[index] = a
        self.rewards[index] = r
        self.terminals[index] = t
        self.current += 1
    
    def sample_memory(self, sample_index):
        pre_states = self.states[sample_index, ...]
        post_states = self.states[sample_index+1, ...]
        actions = self.actions[sample_index]
        rewards = self.rewards[sample_index]
        terminal = self.terminals[sample_index]

        return pre_states, actions, rewards, terminal, post_states
