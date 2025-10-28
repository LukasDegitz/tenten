import numpy as np
import time
from dql import D3QN, ReplayMemory, Transition
from utils import State, Action, Position, random_valid_action, gaussian2d, pieces, transform_state
import torch
import random
import math
import os


class Agent(object):

    target_nets = None
    policy_nets = None

    loss = None
    optimizer = None
    memory = None
    mem_cache = None

    actions_taken = None
    games_played = None #eps
    highscore = None

    # training params
    BETA = None
    BATCH_SIZE = None
    GAMMA = None
    EPS_START = None
    EPS_END = None
    EPS_DECAY = None

    def __init__(self, device='cuda', batch_size=64, gamma=0.99, eps_start=0.9,
                 eps_end=0.05, eps_decay=10000, lr=1e-4, beta=0, tau=0.005):

        self.BETA = beta  #Penalty for a "full board"
        self.TAU = tau
        self.device = device
        # double deep q learning - decouple action value estimation from q value estimation
        self.policy_net = D3QN(device).to(device)
        self.target_net = D3QN(device).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory(1000)
        self.loss = torch.nn.SmoothL1Loss()
        self.mem_cache = {}

        self.actions_taken = 0
        self.games_played = 0
        self.highscore = 0

        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay
        #self.ALPHA = alpha # measures piece vs. position importance
        #self.last_penalty = {'holes':0, 'comps':0, 'centrality':0}
        self.gaussian = gaussian2d()



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
                #print('pred')

                q_hat = self.policy_net(current_state)
                #q_hat = torch.argmax(q_hat, dim=1)[0]
                #action =

        else:
            #print('rand')
            q_hat = torch.randn(current_state.size()[0])
        q_hat = torch.argmax(q_hat, 0)
        q_hat = q_hat.item()
        #print(q_hat)
        return q_hat

    def optimize_model(self):

        #start = time.time()
        if len(self.memory) < self.BATCH_SIZE:
            return

        #torch.autograd.set_detect_anomaly(True)

        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # print(type(batch.state), type(batch.action), type(batch.next_state))
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
        #print(state_action_values.size(), action_offset.size(), action.size())
        state_action_values = state_action_values.gather(0, (action_offset+action).unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_state_pred = self.target_net(next_state)
            next_state_values = torch.zeros(self.BATCH_SIZE).to(self.device)
            non_zero_ids = torch.tensor(tuple(i for i, next_len in enumerate(next_state_lens) if next_len))
            next_state_values[non_zero_ids] = torch.cat(tuple(r.max(0)[0] for r in torch.split(next_state_pred, next_state_lens) if r.size()[0]))
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + rewards

        # Compute Huber loss
        loss = self.loss(state_action_values, expected_state_action_values)
        #print(state_action_value.size(), expected_state_action_value.size())
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        #for param in self.policy_net.parameters():
        #    param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def reward(self, session_score, board, action):

        if session_score < 0:
            return -10
        pops_bonus, _ = divmod(session_score, 10)

        return 1+pops_bonus

        #normalize session score to 1
        #to prevent overestimation of big pieces
        #session_score -= (pieces[action.p_id].sum() - 1)
        # 0 < penalty < beta -> penalize full boards
        #penalty = self.BETA * (board.sum()/100)
        #print(session_score, penalty, session_score * (1 - penalty))
        #return session_score * (1 - penalty) + pops_bonus

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

#TBD
    def init_memory(self, target_path):

        if not os.path.exists(target_path):
            return
        print('initializing memory from: '+target_path)
        i = 0
        for file in os.listdir(target_path):

            file_path = os.path.join(target_path, file)

            with open(file_path, 'r') as r_file:
                for line in r_file:
                    if line == 'act_p_id|act_pos_i|act_pos_j|reward|s_p0|s_p1|s_p2|board_state\n':
                        continue
                    act_p_id, act_pos_i, act_pos_j, reward, s_p0, s_p1, s_p2, board_state = line.split('|')
                    board_state = board_state.replace('\n', '').replace(']', '').replace('[', '')
                    reward, s_p0, s_p1, s_p2 = int(reward), int(s_p0), int(s_p1), int(s_p2)
                    board_state = np.fromstring(board_state, sep=' ').reshape((10, 10))

                    state = make_state(board_state, [s_p0, s_p1, s_p2])
                    state_repr, state_mask = state.repr, state.mask

                    state_repr = state_repr.to(self.device)
                    state_mask = state_mask.to(self.device)

                    # not a starting state
                    if 'reward' in self.mem_cache:
                        self.mem_cache['next_state'] = state_repr
                        self.mem_cache['next_state_mask'] = state_mask
                        self._push_cache()

                    self.mem_cache['state'] = state_repr
                    action = torch.tensor(int(act_p_id+act_pos_i+act_pos_j)).to(self.device)
                    self.mem_cache['action'] = action.unsqueeze(0)
                    self.mem_cache['reward'] = torch.tensor([reward], dtype=torch.float).to(self.device)
                    #print(board_state)

            self.mem_cache = {}

    def save_model(self, path):
        torch.save({
            'epoch': self.games_played,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_state_dict': self.loss.state_dict(),
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.games_played = checkpoint['policy_net_state_dict']
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss.load_state_dict(checkpoint['loss_state_dict'])

