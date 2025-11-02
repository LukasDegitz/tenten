from tkinter import *
from game import Session
from utils import Position, Action, pieces, transform_state, transform_action, wheres
import numpy as np
from agent import Agent

class GameUI(object):

    root = None
    board_grid = None
    board_button_array = None
    piece_grid = None
    score_text = None
    piece_grid_array = None

    agent = None
    visualize_action = None
    reward = None
    next_state = None
    next_mask = None

    session = Session()

    def __init__(self):

        self.session = Session()

        # initialize agent
        agent_cp_path = input('Load agent cp path, enter for None, New for New:')
        self.visualize_action = False
        if not agent_cp_path:
            self.agent = None
        else:
            self.reward = 0.00
            self.next_state = self.session.get_state()
            self.next_mask = self.session.get_mask()

            if agent_cp_path.lower() == 'new':
                self.agent = Agent(device='cuda')
            else:
                self.agent = Agent(device='cuda')
                self.agent.load_model(agent_cp_path)
                self.visualize_action = True

        self.root = Tk()
        self.board_grid = Frame(self.root)
        self.board_grid.grid()
        self.board_button_array = [k for k in range(100)]
        self.piece_button_value = -1

        for t in range(100):
            i, j = divmod(t, 10)
            self.board_button_array[t] = Button(self.board_grid, width=4, height=2,
                                                command=lambda m=t: self.board_button_pressed(m), bg="white")
            #if i+j == 11:
            #    self.board_button_array[t].config(bg = "black")
            self.board_button_array[t].grid(column=j, row=i)
            #self.grid[i, j].config(height = 10, width = 10)

        self.piece_grid = Frame(self.root)
        self.piece_grid.grid()

        self.score_text = Label(self.piece_grid)
        self.score_text.grid(column=1, row=0)
        if self.agent:
            self.score_text.config(text="Score: %i, Reward: %.2f" %(self.session.score, self.reward))
        else:
            self.score_text.config(text="Score: %i" % self.session.score)

        self.piece_button_array = [1, 2, 3]
        self.piece_button_array[0] = Button(self.piece_grid,  width=50, command=lambda: self.piece_button_pressed(0))
        self.piece_button_array[0].grid(column=0, row=1)
        self.piece_button_array[1] = Button(self.piece_grid,  width=50, command=lambda: self.piece_button_pressed(1))
        self.piece_button_array[1].grid(column=1, row=1)
        self.piece_button_array[2] = Button(self.piece_grid,  width=50, command=lambda: self.piece_button_pressed(2))
        self.piece_button_array[2].grid(column=2, row=1)

        self.piece_grid_array = [[r for r in range(25)],
                                 [r for r in range(25)],
                                 [r for r in range(25)]]

        self.piece1_grid = Frame(self.piece_grid)
        self.piece1_grid.grid(column=0, row=2)
        self.piece2_grid = Frame(self.piece_grid)
        self.piece2_grid.grid(column=1, row=2)
        self.piece3_grid = Frame(self.piece_grid)
        self.piece3_grid.grid(column=2, row=2)

        for r in range(25):
            i, j = divmod(r, 5)
            self.piece_grid_array[0][r] = Label(self.piece1_grid, height=2, width=4)
            self.piece_grid_array[0][r].config(bg="white")
            self.piece_grid_array[0][r].grid(column=j, row=i)

            self.piece_grid_array[1][r] = Label(self.piece2_grid, height=2, width=4)
            self.piece_grid_array[1][r].config(bg="white")
            self.piece_grid_array[1][r].grid(column=j, row=i)

            self.piece_grid_array[2][r] = Label(self.piece3_grid, height=2, width=4)
            self.piece_grid_array[2][r].config(bg="white")
            self.piece_grid_array[2][r].grid(column=j, row=i)

        self.update_pieces()
        self.update_board()


    def board_button_pressed(self, t):

        if self.piece_button_value == -1:
            return
        else:

            pos_i, pos_j = divmod(t, 10)
            action = Action(self.session.pieces[self.piece_button_value], Position(pos_i, pos_j))

            # current_state
            current_state = self.next_state
            current_mask = self.next_mask

            step_score = self.session.take_action(action)

            #invalid action
            if step_score == -1:
                return

            if step_score == -1000:
                self.stop()
                return

            # get next state
            self.next_state = self.session.get_state()
            self.next_mask = self.session.get_mask()

            if self.agent:
                self.reward = self.agent.reward(step_score, action, current_state, self.next_state, current_mask, self.next_mask)

            self.update_board()
            self.update_pieces()
            self.update_score()

    def piece_button_pressed(self, t):
        if t >= len(self.session.pieces):
            return
        if self.piece_button_value == -1:#
            self.piece_button_value = t
            self.piece_button_array[t].config(bg='black')

        elif self.piece_button_value == t:
            self.piece_button_value = -1
            self.piece_button_array[t].config(bg='lightgray')

        else:
            self.piece_button_array[self.piece_button_value].config(bg='lightgray')
            self.piece_button_value = t
            self.piece_button_array[t].config(bg='black')

    def update_board(self):

        if self.visualize_action:

            # state representation
            current_state = self.next_state
            current_state_transformed, current_possible_actions = transform_state(current_state)

            # compute action
            q_hat = self.agent.select_action(current_state_transformed)
            agent_action = transform_action(current_possible_actions, q_hat)
            proposed_action_ijs = [[i + agent_action.pos.i, j + agent_action.pos.j]
                                   for i, j in zip(wheres[agent_action.p_id][0], wheres[agent_action.p_id][1])]

        for t in range(100):
            i, j = divmod(t, 10)
            if self.session.board[i, j] == 1:
                self.board_button_array[t].config(bg='black')
            elif self.visualize_action and [i, j] in proposed_action_ijs:
                self.board_button_array[t].config(bg='light blue')
            else:
                self.board_button_array[t].config(bg='white')

    def update_pieces(self):

        for t in range(3):
            if t >= len(self.session.pieces):
                for f in self.piece_grid_array[t]:
                    f.config(bg="white")
            else:
                piece = pieces[self.session.pieces[t]]
                p_size = np.shape(piece)
                for k in range(25):
                    i, j = divmod(k, 5)

                    if i < p_size[0] and j < p_size[1] and piece[i, j] == 1:
                        self.piece_grid_array[t][k].config(bg='black')
                    else:
                        self.piece_grid_array[t][k].config(bg='white')

        self.piece_button_array[self.piece_button_value].config(bg='lightgray')
        self.piece_button_value = -1



    def update_score(self):
        if self.agent:
            self.score_text.config(text="Score: %i, Reward: %.2f" % (self.session.score, self.reward))
        else:
            self.score_text.config(text="Score: %i" % self.session.score)

    def show(self):
        self.root.mainloop()

    def stop(self):
        self.root.destroy()

ui = GameUI()

ui.show()
#time.sleep(30)
#ui.stop()
