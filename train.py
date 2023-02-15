from itertools import count

import numpy as np
import torch
from agent import Agent
from game import Session
from utils import pieces
import time

TARGET_UPDATE = 10
optim_every = 20
log_every = 10
device = 'cuda'
max_eps = 100000

max_score = {'e': -1, 's': 0}

total_score = 0
total_pops = 0
# one agent for now
# basic settings -> try different ones
print('initializing agent')
agent = Agent(device=device)
#agent.init_memory('saves')
start = time.time()
cp_path = 'res/'+time.strftime('%y%m%d_%H%M%S')+'_cp.pt'
save_every = 10000
while agent.games_played < max_eps:

    # initialize game session
    session = Session()
    #play until the session is lost (i.e. no more possible moves)
    for t in count():

        current_state = session.get_state()
        #mask = session.get_mask()
        action = agent.select_action(current_state)
        step_score = session.take_action(action)
        if session.lost:
            losing_state = session.get_state()
            agent.put_reward(step_score, losing_state)

            total_score += session.score
            if session.score > max_score['s']:
                max_score['e'] = agent.games_played
                max_score['s'] = session.score
                print('%i - Max Score: %i  | %i' % (agent.games_played, session.score, t))
            agent.games_played += 1
            break

        if step_score > 10:
            total_pops+=1
        agent.put_reward(step_score)
        if (agent.actions_taken % optim_every == 0):
            agent.optimize_model()


    if agent.games_played % TARGET_UPDATE == 1:
        agent.update_target()

    if agent.games_played % 20 == 1:
        print('EPS %i: AS  %.2f|AA %.2f|AP %.2f| T: %.2f'
              % (agent.games_played, total_score/agent.games_played, agent.actions_taken/agent.games_played,
                 total_pops/agent.games_played, time.time()-start))

    if agent.games_played % save_every == 0:
        torch.save(agent.policy_net.state_dict(), cp_path)

agent.policy_net.eval()
infer_res = {}
print('#' * 50)
f_name = 'res/'+time.strftime('%y%m%d_%H%M%S')+'_inferlog.txt'
with open(f_name, 'w') as w_file:
    print('EPS %i: AS  %.2f|AA %.2f|AP %.2f| T: %.2f'%(agent.games_played, total_score/agent.games_played, agent.actions_taken/agent.games_played,
                 total_pops/agent.games_played, time.time()-start))
    print('writing to: '+f_name)
    print('infer start:')
    w_file.write('Training Result:'+'\n')
    train_res = 'EPS %i: AS  %.2f|AA %.2f|AP %.2f| T: %.2f| MAX_SCORE: %i @ %i eps'%(agent.games_played, total_score/agent.games_played, agent.actions_taken/agent.games_played,
                 total_pops/agent.games_played, time.time()-start, max_score['s'], max_score['e'])+'\n'
    w_file.write(train_res)
    w_file.write('infer start:\n')


    for i in range(10):
        print('#' * 50)
        print('EPS %i' % i)
        w_file.write(('#' * 50) + '\n')
        w_file.write(('EPS %i' % i) + '\n')
        session = Session()
        pops = 0
        start = time.time()
        # play until the session is lost (i.e. no more possible moves)
        for t in count():
            print('-'*30)
            print('step %i'%t)
            w_file.write(('-'*30) + '\n')
            w_file.write(('step %i'%t) + '\n')

            state_str = session.state_str()
            print(state_str)
            w_file.write((state_str) + '\n')

            current_state = session.get_state()
            # mask = session.get_mask()
            action = agent.select_action(current_state)
            print('ACT_PIECE :' )
            print(np.array2string(pieces[action.p_id]))
            print('ACT_POS : %i, %i'%( action.pos.i, action.pos.j))
            w_file.write(('ACT_PIECE :') +'\n')
            w_file.write((np.array2string(pieces[action.p_id]))+'\n')
            w_file.write(('ACT_POS (: %i, %i'%( action.pos.i, action.pos.j))+'\n')
            reward = session.take_action(action)
            if session.lost:
                print('GAME OVER!')
                infer_res[i] = {'score': session.score,
                                'actions': t,
                                'pops': pops,
                                't': round(time.time()-start,2)
                                }
                state_str = session.state_str()
                print(state_str)
                w_file.write((state_str) + '\n')
                agent.games_played += 1
                break

            if reward > 10:
                pops+=1

with open(f_name.replace('inferlog', 'inferres'), 'w') as w_file:
    print('Infer Results:')
    print('RUN|SCORE|ACTIONS|POPS|T')

    w_file.write('Train Result:\n')
    w_file.write(train_res)
    w_file.write('Infer Results:\n')
    w_file.write('RUN|SCORE|ACTIONS|POPS|T\n')
    for i, res in infer_res.items():
        print('%i|%i|%i|%i|%.2f'%(i, res['score'], res['actions'], res['pops'], res['t']))
        w_file.write(('%i|%i|%i|%i|%.2f'%(i, res['score'], res['actions'], res['pops'], res['t']))+'\n')