import torch
from collections import namedtuple, deque
from utils import pieces
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


class stateNN(torch.nn.Module):

    def __init__(self, device):

        super(stateNN, self).__init__()

        self.device = device

        #self.conv = torch.nn.Sequential(
        #    torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5)),
        #    torch.nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #    torch.nn.ReLU(),
        #    torch.nn.MaxPool2d(2, 1),
        #    torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3)),
        #    torch.nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #    torch.nn.ReLU(),
        #    torch.nn.MaxPool2d(2, 1)
        #)


        self.lin = torch.nn.Sequential(
            torch.nn.Linear(100, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 100),
        )

        #self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = x.to(self.device)
        #x = self.conv(x)
        #x = torch.flatten(x, 1)
        return self.lin(x)


class mainNN(torch.nn.Module):

    def __init__(self, device):
        super(mainNN, self).__init__()

        self.device = device

        #self.conv = torch.nn.Sequential(
        #    torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5)),
        #    torch.nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #    torch.nn.ReLU(),
        #    torch.nn.MaxPool2d(2, 1),
        #    torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3)),
        #    torch.nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #    torch.nn.ReLU(),
        #    torch.nn.MaxPool2d(2, 1)
        #)
        self.lin = torch.nn.Sequential(
            torch.nn.Linear(200, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.to(self.device)
       # x = self.conv(x)
       # x = torch.flatten(x, 1)
        return self.lin(x)

class D3QN(torch.nn.Module):

    def __init__(self, device):
        #input is tensor for positions (batchsize x 10 x 10)
        super(D3QN, self).__init__()

        #self.state_net = StateCNN(device=device)
        self.current_stateNN = stateNN(device=device)
        self.action_stateNN = stateNN(device=device)
        self.mainNN = mainNN(device=device)
        self.device = device

    def forward(self, current_state, action_state):
        #expects current_state = (batch_size, n_possible_actions, 10, 10)
        #and actions_state  = (batch_size, n_possible_actions, 10, 10)
        current_state, action_state = current_state.to(self.device), action_state.to(self.device)
        current_state = current_state.reshape((current_state.size()[0], current_state.size()[1], 100))
        action_state = action_state.reshape((action_state.size()[0], action_state.size()[1], 100))

        state_value=self.current_stateNN(current_state)
        action_value=self.action_stateNN(action_state)
        #print(state_value.size(), action_value.size(),torch.cat((state_value, action_value), 2).size())
        q_hat = self.mainNN(torch.cat((state_value, action_value), 2))
        return q_hat
