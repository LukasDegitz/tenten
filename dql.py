import torch
from collections import namedtuple, deque
import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'next_state_mask', 'reward'))

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

class DQN(torch.nn.Module):

    def __init__(self, device):
        #input is tensor for positions (19x10x10) -> 1900 and tensor for pieces 19 => 1919
        super(DQN, self).__init__()

        self.device = device
        # base
        self.ln1 = torch.nn.Linear(in_features=1919, out_features=2048)
        self.bn1 = torch.nn.BatchNorm1d(2048)
        self.ln2 = torch.nn.Linear(in_features=2048, out_features=2048)
        self.bn2 = torch.nn.BatchNorm1d(1024)
        self.ln3 = torch.nn.Linear(in_features=2048, out_features=2048)
        self.bn3 = torch.nn.BatchNorm1d(512)

        #one layer for piece selection and one for position
        #self.l_piece = torch.nn.Linear(in_features=512, out_features=19)
        self.l_pos = torch.nn.Linear(in_features=2048, out_features=1900)

        self.sig = torch.nn.Sigmoid()
        self.rel = torch.nn.ReLU()


    def forward(self, x):
        x = x.to(self.device)
        x = self.rel(self.bn1(self.ln1(x)))
        x = self.rel(self.bn2(self.ln2(x)))
        x = self.rel(self.bn3(self.ln3(x)))

        return self.sig(self.l_pos(x))
