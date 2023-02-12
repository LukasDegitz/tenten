import numpy as np

from dql import D3QN, ReplayMemory, Transition
from utils import State, Action, Position, make_state
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
    BATCH_SIZE = None
    GAMMA = None
    EPS_START = None
    EPS_END = None
    EPS_DECAY = None

    def __init__(self, device='cuda', batch_size=64, gamma=0.99, eps_start=0.9,
                 eps_end=0.05, eps_decay=1000, lr=1e-4, alpha=0.5):

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
        self.ALPHA = alpha

    def select_action(self, state: State):
        sample = random.random()

        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.actions_taken / self.EPS_DECAY)

        self.actions_taken += 1
        state = State(state.pieces.to(self.device), state.board.to(self.device), state.board_mask.to(self.device))

        #not a starting state
        if 'reward' in self.mem_cache:
            self.mem_cache['next_state'] = state
            self._push_cache()

        self.mem_cache['state'] = state

        if sample > eps_threshold:

            with torch.no_grad():
                #print('pred')
                q_piece, q_pos = self.policy_net(state.pieces, state.board, state.board_mask)


        else:
            #print('rand')
            q_piece = torch.sigmoid(torch.randn((1, 19))).to(self.device)
            #q_rand = torch.sigmoid(torch.randn((19, 100))).to(self.device)
            q_piece = q_piece * state.pieces * (state.board_mask.sum(2) >= 1)
            p_id = q_piece.argmax(1)
            q_pos = torch.sigmoid(torch.randn((1, 100))).to(self.device)
            q_pos *= state.board_mask[0, p_id, :]


        p_id = q_piece.argmax(1)
        pos_id = q_pos.argmax(1)
        self.mem_cache['action'] = (p_id.unsqueeze(0), pos_id.unsqueeze(0))

        i, j = divmod(pos_id.item(), 10)
        return Action(p_id.item(), Position(i, j))

    def optimize_model(self):

        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        board_state_batch = torch.cat([state.board for state in batch.state], 0)
        board_mask_batch = torch.cat([state.board_mask for state in batch.state], 0)
        piece_state_batch = torch.cat([state.pieces for state in batch.state], 0)
        piece_action_batch = torch.cat([action[0] for action in batch.action], 0)
        board_action_batch = torch.cat([action[1] for action in batch.action], 0)
        reward_batch = torch.cat(batch.reward, 0)
        next_board_state_batch = torch.cat([state.board for state in batch.next_state], 0)
        next_board_mask_batch = torch.cat([state.board_mask for state in batch.next_state], 0)
        next_piece_state_batch = torch.cat([state.pieces for state in batch.next_state], 0)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        piece_action_values, board_action_values = \
            self.policy_net(piece_state_batch, board_state_batch, board_mask_batch)
        selected_piece_action_values = piece_action_values.gather(1, piece_action_batch)
        selected_board_action_values = board_action_values.gather(1, board_action_batch)
        state_action_values = (selected_piece_action_values * self.ALPHA + selected_board_action_values * (1 - self.ALPHA))

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_piece_state_value, next_board_state_value = \
            self.target_net(next_piece_state_batch, next_board_state_batch, next_board_mask_batch)

        next_piece_state_value, next_board_state_value = \
            next_piece_state_value.max(1)[0].detach(), next_board_state_value.max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = \
            ((next_piece_state_value * self.ALPHA + next_piece_state_value * (1 - self.ALPHA)) * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = self.loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        #for param in self.policy_net.parameters():
        #    param.grad.data.clamp_(-1, 1)
        self.optimizer.step()



    def update_target(self):
        # soft update
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * 0.005 + target_net_state_dict[key] * (1 - 0.005)
        self.target_net.load_state_dict(target_net_state_dict)

    def put_reward(self, reward, losing_state: State = None):

        self.mem_cache['reward'] = torch.tensor([reward], dtype=torch.float).to(self.device)

        if losing_state:
            losing_state = State(losing_state.pieces.to(self.device), losing_state.board.to(self.device),
                                 losing_state.board_mask.to(self.device))
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

