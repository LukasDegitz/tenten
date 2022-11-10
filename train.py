from itertools import count
import torch
from agent import Agent
from game import Session

TARGET_UPDATE = 10
optim_every = 5
device = 'cuda'
max_eps = 500

max_score = {'e': -1, 's': 0}

total_score = 0
# one agent for now
# basic settings -> try different ones
print('initializing agent')
agent = Agent(device=device)
#create a baseline :) -> random moves for idk 1k games or so
while agent.games_played < max_eps:

    # initialize game session
    session = Session()
    #play until the session is lost (i.e. no more possible moves)
    for t in count():

        current_state = session.get_state()
        action = agent.select_action(current_state)

        step_reward = session.take_action(action)
        if session.lost:
            losing_state = session.get_state()
            agent.put_reward(step_reward, losing_state)

            total_score += session.score
            if session.score > max_score['s']:
                max_score['e'] = agent.games_played
                max_score['s'] = session.score
                print('Max Score - %i: %i (%i)' % (agent.games_played, session.score, t))
            agent.games_played += 1
            break

        agent.put_reward(step_reward)
        if (agent.actions_taken % optim_every == 0):
            agent.optimize_model()

    if agent.games_played % TARGET_UPDATE == 1:
        agent.update_target()
        print('%i: AS  %.2f|AA %.2f' % (agent.games_played, total_score/agent.games_played, agent.actions_taken/agent.games_played))
