from dql import D3QN, ReplayMemory, Transition
from utils import gaussian2d, sums, count_corners
import torch
import random
import math
import numpy as np
from scipy.ndimage import label

class Agent(object):

    #DQN components
    target_net = None
    policy_net = None
    loss = None
    optimizer = None
    memory = None

    actions_taken = None
    games_played = None #eps

    # penalty params
    ALPHA = None
    BETA = None
    GAMMA = None
    DELTA = None
    EPSILON = None

    # training params
    BATCH_SIZE = None
    RHO = None
    EPS_START = None
    EPS_END = None
    EPS_DECAY = None

    def __init__(self, device='cuda', batch_size=64, rho=0.99, eps_start=0.9,
                 eps_end=0.05, eps_decay=10000, lr=3e-4, tau=0.005,
                 alpha=1, beta=1, gamma=0.4, delta=1, epsilon=1):
        #lr for small 1e-4, lr for beeg 3e-4

        self.ALPHA = alpha #Penalty for a "full board"
        self.BETA = beta  #full board trade of between placability and centeredness
        self.GAMMA = gamma
        self.DELTA = delta
        self.EPSILON = epsilon
        self.TAU = tau

        self.device = device

        # double deep q learning - decouple action value estimation from q value estimation
        self.policy_net = D3QN(device).to(device)
        self.target_net = D3QN(device).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.loss = torch.nn.SmoothL1Loss()

        self.actions_taken = 0
        self.games_played = 0

        self.BATCH_SIZE = batch_size
        self.RHO = rho
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay


    def select_action(self, current_state, mode='train'):

        sample = random.random()
        if mode == 'infer':
            eps_threshold = 0.000
        else:
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.actions_taken / self.EPS_DECAY)

        self.actions_taken += 1

        if sample > eps_threshold:
            with torch.no_grad():
                q_hat = self.policy_net(current_state)
        else:
            q_hat = torch.randn(current_state.size()[0])

        q_hat = torch.argmax(q_hat, 0)
        q_hat = q_hat.item()
        #print(q_hat)
        return q_hat

    def optimize_model(self):

        #start = time.time()
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        rewards = torch.cat(batch.reward).to(self.device)
        state = torch.cat(batch.state).to(self.device)
        action = torch.cat(batch.action).to(self.device)
        next_state = torch.cat(batch.next_state).to(self.device)

        #get lengths of each batch items to align predictions and rewards later
        state_lens = torch.tensor(tuple(state.size()[0] for state in batch.state)).to(self.device)
        next_state_lens = tuple(next_state.size()[0] for next_state in batch.next_state)

        #compute offset to gather actions
        action_offset = torch.roll(state_lens, 1)
        action_offset[0] = 0
        action_offset = torch.cumsum(action_offset, 0)

        state_action_values = self.policy_net(state)
        state_action_values = state_action_values.gather(0, (action_offset+action).unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_state_pred = self.target_net(next_state)
            next_state_values = torch.zeros(self.BATCH_SIZE).to(self.device)
            non_zero_ids = torch.tensor(tuple(i for i, next_len in enumerate(next_state_lens) if next_len))
            next_state_values[non_zero_ids] = torch.cat(tuple(r.max(0)[0] for r in torch.split(next_state_pred, next_state_lens) if r.size()[0]))

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.RHO) + rewards

        # Compute Huber loss
        loss = self.loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        #for param in self.policy_net.parameters():
        #    param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def reward(self, session_score, action, current_state, next_state, current_mask, next_mask):

        if session_score < 0:
            return -10


        #pen_a: change in piece placeability  (0-1 -> low to high penalisation)
        current_placeability = ((current_mask.sum(axis=(1, 2)) * sums).sum())
        next_placeability = ((next_mask.sum(axis=(1, 2)) * sums).sum())
        pen_a = 4 * (current_placeability - next_placeability)/current_placeability

        # pen_b number of holes
        _, current_holes = label(np.logical_not(current_state.board).astype(int))
        _, next_holes = label(np.logical_not(next_state.board).astype(int))
        pen_b = (next_holes - current_holes) / max(((8-current_holes), 2))

        # pen_c: change in board center occupation (0-1 -> low to high penalisation)
        current_centricity = (current_state.board*gaussian2d).sum()
        next_centricity = (next_state.board*gaussian2d).sum()
        #print('current: %.2f, next: %.2f' % (current_centricity, next_centricity))
        pen_c = (next_centricity - current_centricity)/(sums[action.p_id])

        #pen_d number of corners
        current_corners = count_corners(current_state.board)
        next_corners = count_corners(next_state.board)
        pen_d = (next_corners - current_corners) / max(((20-current_corners), 6))

        #pend_e number of isles
        _, current_isles = label(current_state.board)
        _, next_isles = label(next_state.board)
        pen_e = (next_isles - current_isles) / max(((12 - current_isles), 2))

        # penalize biggest penalty of all three full, and parameterized average of others
        penalty = (self.ALPHA * pen_a) + (self.BETA * pen_b) + (self.GAMMA * pen_c) + (self.DELTA * pen_d) \
                  + (self.EPSILON * pen_e)

        #print('%i, a: %.2f, b: %.2f,c: %.2f, d:%.2f e:%.2f, pen: %.2f | rew: %.2f'
        #      %(session_score, pen_a, pen_b, pen_c, pen_d, pen_e, penalty, 1-penalty))

        return 1 - penalty

    def update_target(self):
        # soft update

        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (1 - self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

    def push_batch(self, reward, state, action, next_state):

        reward = torch.tensor([reward], dtype=torch.float).to(self.device)
        self.memory.push(state, torch.tensor(action).unsqueeze(0), next_state, reward)


    def save_model(self, path):
        torch.save({
            'epoch': self.games_played,
            'actions_taken': self.actions_taken,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_state_dict': self.loss.state_dict(),
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.games_played = checkpoint['epoch']
        self.actions_taken = checkpoint['actions_taken']
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss.load_state_dict(checkpoint['loss_state_dict'])


