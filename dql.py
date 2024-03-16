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

        self.softmax = torch.nn.Softmax(dim=1)

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
            torch.nn.Linear(128, 1),
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

    def __init__(self, device, piece):
        #input is tensor for positions (batchsize x 10 x 10)
        super(D3QN, self).__init__()

        self.state_net = StateCNN(device=device)
        self.act_net = ActCNN(device=device)
        self.device = device
        self.piece = piece.unsqueeze(0).unsqueeze(0).to(self.device)

    def forward(self, board):
        #expects board = (batch_size,1, 10,10) and piece(batch_size

        board = board.to(self.device)
        #transform board
        board = board.unsqueeze(1)
        board = torch.nn.functional.conv2d(board, self.piece)/torch.sum(self.piece, (2, 3))
        #pad zeros to retain shape
        #print(board.shape)
        board = torch.nn.functional.pad(board, (0, 10-board.shape[3], 0, 10-board.shape[2]))
        board_state, board_act = torch.clone(board), torch.clone(board)
        # estimated q_value, masked by the list of available pieces and whether a piece can be placed
        #print(board.shape)
        v_state = self.state_net(board_state)
        v_action = self.act_net(board_act)
        #v_action = torch.cat([self.act_net[p_id](x_board[:, p_id, :, :].unsqueeze(1)).unsqueeze(1)
        #                      for p_id in range(19)], 1)
        # Don't know why we normalize here again.
        q = v_state + (v_action - v_action.mean(1).unsqueeze(-1))
        q = torch.square(q).reshape((q.size()[0], 10, 10))

        #mask illegal values with board
        board[board < 1] = 0
        return q * board.squeeze(1)
