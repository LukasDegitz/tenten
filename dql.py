import torch
from collections import namedtuple, deque
from utils import base_position_mask, sigmoid_mask
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


class ActCNN(torch.nn.Module):

    def __init__(self, device):

        super(ActCNN, self).__init__()

        self.device = device

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5, 5)),
            torch.nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 1),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3)),
            torch.nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 1)
        )
        self.lin = torch.nn.Sequential(
            torch.nn.Linear(32 * 2 * 2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 100),
        )

    def forward(self, x):
        x = x.to(self.device)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.lin(x)

class StateCNN(torch.nn.Module):

    def __init__(self, device):
        super(StateCNN, self).__init__()

        self.device = device

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=19, out_channels=32, kernel_size=(5, 5)),
            torch.nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 1),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
            torch.nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 1)
        )
        self.lin = torch.nn.Sequential(
            torch.nn.Linear(64 * 2 * 2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 19),
        )

    def forward(self, x):
        x = x.to(self.device)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.lin(x)

class D3QN(torch.nn.Module):

    #-> The problem with this architecture is, that is might predict the value of an action very precisely
    # But disregards the value of the current (-> the prediction the future) state of the board
    # So we introduce a second MLP to estimate the board state
    def __init__(self, device):
        #input is tensor for positions (19x10x10) -> 1900 and tensor for pieces 19 => 1919
        super(D3QN, self).__init__()

        self.state_net = StateCNN(device=device)
        self.act_net = torch.nn.ModuleList([ActCNN(device=device) for _ in range(19)])
        self.device = device
        #self.softmax = torch.softmax


    def forward(self, x_board, x_board_mask):
        x_board, x_board_mask = x_board.to(self.device), x_board_mask.to(self.device)
        # estimated q_value, masked by the list of available pieces and whether a piece can be placed

        v_state = self.state_net(x_board)
        v_action = torch.cat([self.act_net[p_id](x_board[:, p_id, :, :].unsqueeze(1)).unsqueeze(1)
                              for p_id in range(19)], 1)
        q = v_state.unsqueeze(-1) + (v_action - v_action.mean(2).unsqueeze(-1))
        q_masked = torch.square(q).reshape((q.size()[0], 19, 10, 10)) * x_board_mask
        #print(q_state, q_pieces.max(1)[0], q_board.max(1)[0])
        return q_masked
