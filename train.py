from itertools import count

import numpy as np
import torch
from agent import Agent
from game import Session
from utils import pieces, transform_action
import time

TARGET_UPDATE = 50
optim_every = 4
log_every = 10
device = 'cuda'
max_eps = 10000

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
train_log_path = 'res/'+time.strftime('%y%m%d_%H%M%S')+'_trainlog.txt'
save_every = 500
while agent.games_played < max_eps:

    # initialize game session
    session = Session()
    next_state_transformed, next_possible_actions = session.get_transformed_state()
    #play until the session is lost (i.e. no more possible moves)
    for t in count():

        current_state_transformed, current_possible_actions = next_state_transformed, next_possible_actions

        q_hat = agent.select_action(current_state_transformed)
        action = transform_action(current_possible_actions, q_hat)
        step_score = session.take_action(action)
        next_state_transformed, next_possible_actions = session.get_transformed_state()
        rewards = agent.reward(step_score, session.get_mask(), action)

        agent.push_batch(step_score, state=current_state_transformed, action=q_hat, next_state=next_state_transformed)
        if session.lost:

            total_score += session.score
            if session.score > max_score['s']:
                max_score['e'] = agent.games_played
                max_score['s'] = session.score
                print('%i - Max Score: %i  | %i' % (agent.games_played, session.score, t))
                with open(train_log_path, 'a') as train_log_file:
                    train_log_file.write(('EPS %i: Max Score: %i  | Actions: %i' % (agent.games_played, session.score, t))+'\n')
            agent.games_played += 1
            break

        if step_score > 10:
            total_pops+=1

    if (agent.games_played % optim_every == 1):
        agent.optimize_model()


    if agent.games_played % TARGET_UPDATE == 1:
        agent.update_target()

    if agent.games_played % 20 == 1:
        print('EPS %i: AS  %.2f|AA %.2f|AP %.2f| T: %.2f'
              % (agent.games_played, total_score/agent.games_played, agent.actions_taken/agent.games_played,
                 total_pops/agent.games_played, time.time()-start))
        with open(train_log_path, 'a') as train_log_file:
            train_log_file.write(('EPS %i: AS  %.2f|AA %.2f|AP %.2f| T: %.2f'
              % (agent.games_played, total_score/agent.games_played, agent.actions_taken/agent.games_played,
                 total_pops/agent.games_played, time.time()-start))+'\n')

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
                #print('#' * 50)
                #print('EPS %i' % i)
                w_file.write(('#' * 50) + '\n')
                w_file.write(('EPS %i' % i) + '\n')
                session = Session()
                pops = 0
                start = time.time()
                # play until the session is lost (i.e. no more possible moves)
                for t in count():
                    #print('-'*30)
                    #print('step %i'%t)
                    w_file.write(('-'*30) + '\n')
                    w_file.write(('step %i'%t) + '\n')

                    state_str = session.state_str()
                    #print(state_str)
                    w_file.write((state_str) + '\n')

                    current_state_transformed, current_possible_actions  = session.get_transformed_state()
                    q_hat = agent.select_action(current_state_transformed, mode='infer')
                    # mask = session.get_mask()
                    action = transform_action(current_possible_actions, q_hat)

                    #print('ACT_PIECE :' )
                    #print(np.array2string(pieces[action.p_id]))
                    #print('ACT_POS : %i, %i'%( action.pos.i, action.pos.j))
                    w_file.write(('ACT_PIECE :') +'\n')
                    w_file.write((np.array2string(pieces[action.p_id]))+'\n')
                    w_file.write(('ACT_POS (: %i, %i'%( action.pos.i, action.pos.j))+'\n')
                    reward = session.take_action(action)
                    if session.lost:
                        #print('GAME OVER!')
                        infer_res[i] = {'score': session.score,
                                        'actions': t,
                                        'pops': pops,
                                        't': round(time.time()-start,2)
                                        }
                        state_str = session.state_str()
                        #print(state_str)
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
            inf_score, inf_act, inf_pops = [],[],[]
            for i, res in infer_res.items():
                print('%i|%i|%i|%i|%.2f'%(i, res['score'], res['actions'], res['pops'], res['t']))
                w_file.write(('%i|%i|%i|%i|%.2f'%(i, res['score'], res['actions'], res['pops'], res['t']))+'\n')
                inf_score.append(res['score'])
                inf_act.append(res['actions'])
                inf_pops.append(res['pops'])
            print('AVGS - SCORE: %.2f|ACTS: %.2f|POPS: %.2f'%(sum(inf_score)/len(inf_score), sum(inf_act)/len(inf_act), sum(inf_pops)/len(inf_pops)))
            w_file.write(('AVGS - SCORE: %.2f|ACTS: %.2f|POPS: %.2f' % (
            sum(inf_score) / len(inf_score), sum(inf_act) / len(inf_act), sum(inf_pops) / len(inf_pops))))
            with open(train_log_path, 'a') as train_log_file:
                train_log_file.write(('INFER AVG - SCORE: %.2f|ACTS: %.2f|POPS: %.2f'%(sum(inf_score)/len(inf_score), sum(inf_act)/len(inf_act), sum(inf_pops)/len(inf_pops)))+'\n')