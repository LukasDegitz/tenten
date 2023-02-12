from itertools import count
from agent import Agent
from game import Session
import math

TARGET_UPDATE = 5
optim_every = 10
device = 'cuda'
max_eps = 100000

max_score = {'e': -1, 's': 0}

total_score = 0
n_pops = 0
# one agent for now
# basic settings -> try different ones
print('initializing agent')
agent = Agent(device=device)
#agent.init_memory('saves')
#create a baseline :) -> random moves for idk 1k games or so
while agent.games_played < max_eps:

    # initialize game session
    session = Session()
    #play until the session is lost (i.e. no more possible moves)
    for t in count():

        current_state = session.get_state()
        #mask = session.get_mask()
        action = agent.select_action(current_state)

        step_reward = session.take_action(action)
        if session.lost:
            losing_state = session.get_state()
            agent.put_reward(step_reward, losing_state)

            total_score += session.score
            if session.score > max_score['s']:
                max_score['e'] = agent.games_played
                max_score['s'] = session.score
                print('%i - Max Score: %i  | %i' % (agent.games_played, session.score, t))
            agent.games_played += 1
            break

        if step_reward > 10:
            n_pops+=1
        agent.put_reward(step_reward)
        if (agent.actions_taken % optim_every == 0):
            agent.optimize_model()


    if agent.games_played % TARGET_UPDATE == 1:

        agent.update_target()
        print('%i: AS  %.2f|AA %.2f|POP %i'
              % (agent.games_played, total_score/agent.games_played, agent.actions_taken/agent.games_played,
                 n_pops))
        n_pops = 0
