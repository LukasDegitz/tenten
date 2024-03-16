import numpy as np
import time
from dql import D3QN, ReplayMemory, Transition
from utils import State, Action, Position, random_valid_action, gaussian2d, pieces
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
                 eps_end=0.05, eps_decay=1000, lr=1e-4, beta=0.5, tau=0.005):

        self.BETA = beta  #Penalty for a "full board"
        self.TAU = tau
        self.device = device
        # double deep q learning - decouple action value estimation from q value estimation
        #One net per piece
        self.policy_nets = torch.nn.ModuleList()
        self.target_nets = torch.nn.ModuleList()
        self.optimizers = []
        for i, piece in enumerate(pieces):
            self.policy_nets.append(D3QN(device, torch.tensor(piece, dtype=torch.float)).to(device))
            self.target_nets.append(D3QN(device, torch.tensor(piece, dtype=torch.float)).to(device))
            self.target_nets[i].load_state_dict(self.policy_nets[i].state_dict())
            self.target_nets[i].eval()
            self.optimizers.append(torch.optim.AdamW(self.policy_nets[i].parameters(), lr=lr, amsgrad=True))

        self.loss = torch.nn.SmoothL1Loss()
        self.memory = [ReplayMemory(1000) for _ in range(19)]
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


    def select_action(self, state: State, mode='train'):
        sample = random.random()
        if mode == 'infer':
            eps_threshold = 0.000
        else:
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.actions_taken / self.EPS_DECAY)

        self.actions_taken += 1
        #state = State(state.board.to(self.device), state.mask.to(self.device))

        #not a starting state
        #if 'reward' in self.mem_cache:
        #    self.mem_cache['next_state'] = state
        #    self._push_cache()

        self.mem_cache['state'] = state

        if sample > eps_threshold:

            with torch.no_grad():
                #print('pred')
                #invert board and make it a tensor
                board = np.logical_not(state.board).astype(int)
                board = torch.tensor(board, dtype=torch.float).unsqueeze(0)

                q_max = 0
                act_pos_max = 0
                act_piece_max = state.pieces[0]
                for piece in set(state.pieces): #no need to duplicate

                    q_hat = self.policy_nets[piece](board)
                    q_hat, act_pos = q_hat.reshape(q_hat.shape[0], 100).max(1)
                    q_hat, act_pos = q_hat.item(), act_pos.item()

                    if q_hat > q_max:
                        q_max, act_pos_max, act_piece_max = q_hat, act_pos, piece

                #print(q_max)
                act_i, act_j = divmod(act_pos_max, 10)
                action = Action(act_piece_max, Position(act_i, act_j))

        else:
            #print('rand')
            action = random_valid_action(state=state)

        self.mem_cache['action'] = action

        return action

    def optimize_model(self):

        #start = time.time()
        if min([len(piece_mem) for piece_mem in self.memory]) < self.BATCH_SIZE:
            return

        #torch.autograd.set_detect_anomaly(True)
        for piece_mem, policy_net, target_net, optimizer in zip(self.memory, self.policy_nets, self.target_nets, self.optimizers):
            transitions = piece_mem.sample(self.BATCH_SIZE)
            # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
            # detailed explanation). This converts batch-array of Transitions
            # to Transition of batch-arrays.
            batch = Transition(*zip(*transitions))

            state_board_batch = np.array([np.logical_not(state.board).astype(int) for state in batch.state])
            state_board_batch = torch.tensor(state_board_batch, dtype=torch.float).to(self.device)


            action_batch = torch.tensor([(10 * action.pos.i) + action.pos.j for action in batch.action]).to(self.device)
            action_batch = action_batch.unsqueeze(-1)
            reward_batch = torch.cat(batch.reward, 0).to(self.device)

            next_state_board_batch = np.array([np.logical_not(next_state.board).astype(int)
                                               for next_state in batch.next_state])

            next_state_board_batch = torch.tensor(next_state_board_batch, dtype=torch.float).to(self.device)

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_action_values = policy_net(state_board_batch)
            state_action_values = state_action_values.reshape((state_action_values.size()[0], 100))
            state_action_values = state_action_values.gather(1, action_batch).squeeze(1)
            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1)[0].
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.

            with torch.no_grad():
                next_state_value = target_net(next_state_board_batch)
                next_state_value = next_state_value.reshape((next_state_value.size()[0], 100)).max(1)[0].detach()

            # Compute the expected Q values

            expected_state_action_values = (next_state_value * self.GAMMA) + reward_batch
            # Compute Huber loss

            loss = self.loss(state_action_values, expected_state_action_values)

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()

            #torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
            for param in policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()

        #print('optimizer time: %.2f s'%(time.time()-start))

    def reward(self, session_score, action: Action, board):

        #normalize session score to 1
        #to prevent overestimation of big pieces
        session_score -= (pieces[action.p_id].sum() + 1)
        # 0 < penalty < beta -> penalize full boards
        penalty = self.BETA * (board.sum()/100)
        return session_score * (1 - penalty)

    def update_target(self):
        # soft update
        for target_net, policy_net in zip(self.target_nets, self.policy_nets):
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (1 - self.TAU)
            target_net.load_state_dict(target_net_state_dict)

    def put_reward(self, step_score, state: State, action: Action, next_state: State):

        reward = self.reward(step_score, action, next_state.board)
        reward = torch.tensor([reward], dtype=torch.float).to(self.device)

        self.memory[action.p_id].push(state, action, next_state, reward)

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

