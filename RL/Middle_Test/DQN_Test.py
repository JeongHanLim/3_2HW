import gym
import collections
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

learning_rate = 0.05
gamma = 0.99
buffer_limit = 50000
batch_size = 32

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_list, a_list, r_list, sprime_list, donemask_list = [],[],[],[],[]
        for transition in mini_batch:
            s, a, r, sprime, donemask = transition
            s_list.append(s)
            a_list.append([a])
            r_list.append([r])
            sprime_list.append(sprime)
            donemask_list.append([donemask])

        return torch.tensor(s_list, dtype = torch.float), torch.tensor(a_list), \
               torch.tensor(r_list), torch.tensor(sprime_list, dtype = torch.float), \
               torch.tensor(donemask_list)

    def size(self):
        return len(self.buffer)



