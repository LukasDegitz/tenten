from dql import DQN, ReplayMemory, Transition
from utils import State, transform_state, parse_action
import torch
import random
import math



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
    BATCH_SIZE = None
    GAMMA = None
    EPS_START = None
    EPS_END = None
    EPS_DECAY = None

    def __init__(self, device, batch_size=128, gamma=0.999, eps_start=0.9, eps_end=0.05, eps_decay=200):

        self.device = device
        self.policy_net = DQN(device).to(device)
        self.target_net = DQN(device).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.loss = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters())
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

    def select_action(self, state: State):
        sample = random.random()

        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.actions_taken / self.EPS_DECAY)

        self.actions_taken += 1

        state_t, state_m = transform_state(state)
        state_t = state_t.to(self.device)
        state_m = state_m.to(self.device)

        #not a starting state
        if 'reward' in self.mem_cache:
            self.mem_cache['next_state'] = state_t
            self.mem_cache['next_state_mask'] = state_m
            self._push_cache()

        self.mem_cache['state'] = state_t

        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.

                p_pos = self.policy_net(state_t)
        else:
            p_pos = torch.randn(1900).to(self.device)

        p_pos_masked = p_pos * state_m
        action = torch.argmax(p_pos_masked)
        self.mem_cache['action'] = action.unsqueeze(0)
        return parse_action(action)

    def optimize_model(self):

        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        next_state_mask_batch = torch.cat(batch.next_state_mask)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = (self.target_net(next_state_batch) * next_state_mask_batch).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = self.loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def put_reward(self, reward, losing_state: State = None):

        self.mem_cache['reward'] = torch.tensor([reward], dtype=torch.float).to(self.device)

        if losing_state:
            state_t, state_m = transform_state(losing_state)
            state_t = state_t.to(self.device)
            state_m = state_m.to(self.device)
            self.mem_cache['next_state'] = state_t
            self.mem_cache['next_state_mask'] = state_m
            self._push_cache()

    def _push_cache(self):
        if len(self.mem_cache) != 5:
            print('Memory error, clearing cache. Memory of step is lost!')
            self.mem_cache = {}
            return

        self.memory.push(self.mem_cache['state'], self.mem_cache['action'], self.mem_cache['next_state'],
                         self.mem_cache['next_state_mask'], self.mem_cache['reward'])
        self.mem_cache = {}