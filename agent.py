import numpy as np

from dql import D3QN, ReplayMemory, Transition
from utils import State, Action, Position, make_state, sigmoid_mask
import torch
import random
import math
import os


class Agent(object):

    target_net = None
    policy_net = None

    loss = None
    optimizer = None
    memory = None
    mem_cache = None

    actions_taken = None
    games_played = None #eps
    highscore = None

    # training params
    ALPHA = None
    BETA = None
    BATCH_SIZE = None
    GAMMA = None
    EPS_START = None
    EPS_END = None
    EPS_DECAY = None

    def __init__(self, device='cuda', batch_size=128, gamma=0.99, eps_start=0.9,
                 eps_end=0.05, eps_decay=3000, lr=5e-4, alpha=0, beta=0.5):

        self.BETA = beta  # measures state vs. action importance
        self.device = device
        # double deep q learning - decouple action value estimation from q value estimation
        self.policy_net = D3QN(device).to(device)
        self.target_net = D3QN(device).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.loss = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.mem_cache = {}

        self.actions_taken = 0
        self.games_played = 0
        self.highscore = 0

        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay
        self.ALPHA = alpha # measures piece vs. position importance


    def select_action(self, state: State, mode='train'):
        sample = random.random()
        if mode == 'infer':
            eps_threshold = 0.000
        else:
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.actions_taken / self.EPS_DECAY)

        self.actions_taken += 1
        state = State(state.board.to(self.device), state.mask.to(self.device))

        #not a starting state
        if 'reward' in self.mem_cache:
            self.mem_cache['next_state'] = state
            self._push_cache()

        self.mem_cache['state'] = state

        if sample > eps_threshold:

            with torch.no_grad():
                #print('pred')
                q_hat = self.policy_net(state.board, state.mask)


        else:
            #print('rand')
            q_rand = torch.square(torch.rand((1, 19, 10, 10))).to(self.device)
            q_hat = q_rand*state.mask

        q_hat = q_hat.reshape((1, 1900))
        q_max, act = q_hat.max(1)
        self.mem_cache['action'] = act.unsqueeze(0)

        p_id, pos = divmod(act.item(), 100)
        i, j = divmod(pos, 10)
        return Action(p_id, Position(i, j))

    def optimize_model(self):

        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        state_board_batch = torch.cat([state.board for state in batch.state], 0)
        state_mask_batch = torch.cat([state.mask for state in batch.state], 0)
        action_batch = torch.cat(batch.action, 0)
        reward_batch = torch.cat(batch.reward, 0)
        next_state_board_batch = torch.cat([state.board for state in batch.next_state], 0)
        next_state_mask_batch = torch.cat([state.mask for state in batch.next_state], 0)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_board_batch, state_mask_batch)
        state_action_values = state_action_values.reshape((state_action_values.size()[0], 1900))
        state_action_values = state_action_values.gather(1, action_batch).squeeze(1)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.

        next_state_value = self.target_net(next_state_board_batch, next_state_mask_batch)
        next_state_value = next_state_value.reshape((next_state_value.size()[0], 1900)).max(1)[0].detach()

        # Compute the expected Q values

        expected_state_action_values = (next_state_value * self.GAMMA) + reward_batch
        # Compute Huber loss
        loss = self.loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        #for param in self.policy_net.parameters():
        #    param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def reward(self, session_score):

        #print(torch.softmax(session_score, dim=0))
        return session_score

    def update_target(self):
        # soft update
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * 0.009 + target_net_state_dict[key] * (1 - 0.005)
        self.target_net.load_state_dict(target_net_state_dict)

    def put_reward(self, step_score, losing_state: State = None):

        score_tensor = torch.tensor([step_score], dtype=torch.float).to(self.device)
        reward = self.reward(score_tensor)
        self.mem_cache['reward'] = reward

        if losing_state:
            losing_state = State(losing_state.board.to(self.device), losing_state.mask.to(self.device))
            self.mem_cache['next_state'] = losing_state
            self._push_cache()

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


    def _push_cache(self):
        if len(self.mem_cache) != 4:
            print('Memory error, clearing cache. Memory of step is lost!')
            self.mem_cache = {}
            return

        self.memory.push(self.mem_cache['state'], self.mem_cache['action'], self.mem_cache['next_state']
                         , self.mem_cache['reward'])
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

