import torch
from collections import namedtuple, deque
import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class mainNN(torch.nn.Module):

    def __init__(self, device):
        super(mainNN, self).__init__()

        self.device = device

        self.lin = torch.nn.Sequential(
            torch.nn.Linear(219, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

        #self.lin = torch.nn.Sequential(
        #    torch.nn.Linear(2138, 4096),
        #    torch.nn.ReLU(),
        #    torch.nn.Linear(4096, 1024),
        #    torch.nn.ReLU(),
        #    torch.nn.Linear(1024, 256),
        #    torch.nn.ReLU(),
        #    torch.nn.Linear(256, 128),
        #    torch.nn.ReLU(),
        #    torch.nn.Linear(128, 64),
        #    torch.nn.ReLU(),
        #    torch.nn.Linear(64, 1)
        #)

    def forward(self, x):
        x = x.to(self.device)
       # x = self.conv(x)
       # x = torch.flatten(x, 1)
        return self.lin(x)

class D3QN(torch.nn.Module):

    def __init__(self, device):


        super(D3QN, self).__init__()

        self.mainNN = mainNN(device=device)
        self.device = device

    def forward(self, current_state):

        current_state = current_state.to(self.device)
        res = self.mainNN(current_state)

        return res
