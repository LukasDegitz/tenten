import torch
from collections import namedtuple, deque
from utils import base_position_mask, State
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


class CNN(torch.nn.Module):

    def __init__(self, device, out_dim):
        super(CNN, self).__init__()

        self.device = device

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=19, out_channels=32, kernel_size=(5, 5)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 1),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 1),
        )
        self.lin = torch.nn.Sequential(
            torch.nn.Linear(64*2*2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, out_dim),
            torch.nn.Softmax(dim=0)
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

        self.device = device
        self.base_position_tensor = torch.tensor(base_position_mask).reshape((19, 100)).to(device)

        self.state_net = CNN(device=device, out_dim=1)
        self.piece_net = CNN(device=device, out_dim=19)
        self.position_net = torch.nn.ModuleList([CNN(device=device, out_dim=100) for _ in range(19)])


    def forward(self, x_pieces, x_board, x_board_mask):
        x_board, x_pieces, x_board_mask = x_board.to(self.device), x_pieces.to(self.device), x_board_mask.to(self.device)
        q_state = self.state_net(x_board)
        q_pieces = self.piece_net(x_board) * x_pieces * q_state * (x_board_mask.sum(2) >= 1)
        x_piece_idx = q_pieces.argmax(1)
        q_board = torch.cat([self.position_net[p_id](x_board[batch].unsqueeze(0)) *
                             x_board_mask[batch, p_id, :] for batch, p_id in enumerate(x_piece_idx)], 0)

        return q_pieces, q_board*q_state
