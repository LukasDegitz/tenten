from tkinter import *
from game import Session
from utils import Position, Action, pieces
import numpy as np
import os

class GameUI(object):

    root = None
    board_grid = None
    board_button_array = None
    piece_grid = None
    score_text = None
    piece_grid_array = None

    save_file = None
    session = Session()

    def __init__(self):

        self.session = Session()

        self.root = Tk()
        self.board_grid = Frame(self.root)
        self.board_grid.grid()
        self.board_button_array = [k for k in range(100)]
        self.piece_button_value = -1

        for t in range(100):
            i, j = divmod(t, 10)
            self.board_button_array[t] = Button(self.board_grid, width=2, height=1,
                                                command=lambda m=t: self.board_button_pressed(m), bg="white")
            #if i+j == 11:
            #    self.board_button_array[t].config(bg = "black")
            self.board_button_array[t].grid(column=j, row=i)
            #self.grid[i, j].config(height = 10, width = 10)

        self.piece_grid = Frame(self.root)
        self.piece_grid.grid()

        self.score_text = Label(self.piece_grid)
        self.score_text.grid(column=1, row=0)
        self.score_text.config(text="Score: %i" % self.session.score)

        self.piece_button_array = [1, 2, 3]
        self.piece_button_array[0] = Button(self.piece_grid,  width=25, command=lambda: self.piece_button_pressed(0))
        self.piece_button_array[0].grid(column=0, row=1)
        self.piece_button_array[1] = Button(self.piece_grid,  width=25, command=lambda: self.piece_button_pressed(1))
        self.piece_button_array[1].grid(column=1, row=1)
        self.piece_button_array[2] = Button(self.piece_grid,  width=25, command=lambda: self.piece_button_pressed(2))
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
            self.piece_grid_array[0][r] = Label(self.piece1_grid, height=1, width=2)
            self.piece_grid_array[0][r].config(bg="white")
            self.piece_grid_array[0][r].grid(column=j, row=i)

            self.piece_grid_array[1][r] = Label(self.piece2_grid, height=1, width=2)
            self.piece_grid_array[1][r].config(bg="white")
            self.piece_grid_array[1][r].grid(column=j, row=i)

            self.piece_grid_array[2][r] = Label(self.piece3_grid, height=1, width=2)
            self.piece_grid_array[2][r].config(bg="white")
            self.piece_grid_array[2][r].grid(column=j, row=i)

        self.update_pieces(self.session.pieces)

        for k in range(1000):
            pot_file = 'saves/save%i.csv' % k
            if os.path.exists(pot_file):
                continue
            else:
                self.save_file = pot_file
                with open(self.save_file, 'w') as f:
                    f.write('act_p_id|act_pos_i|act_pos_j|reward|s_p0|s_p1|s_p2|board_state\n')

                break

    def board_button_pressed(self, t):

        if self.piece_button_value == -1:
            return
        else:
            pos_i, pos_j = divmod(t, 10)
            action = Action(self.session.pieces[self.piece_button_value], Position(pos_i, pos_j))
            board_state = self.session.board.copy()
            pieces_state = self.session.pieces.copy()
            step_score = self.session.take_action(action)
            if step_score < 0:
                if step_score == -1000:
                    self.write_action(action, step_score, pieces_state, board_state)
                    self.write_action(action, 0, self.session.pieces, self.session.board)
                    self.stop()
                return
            self.update_board(self.session.board)
            self.update_pieces(self.session.pieces)
            self.update_score(self.session.score)
            self.write_action(action, step_score,pieces_state, board_state)

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

    def update_board(self, board):
        for t in range(100):
            i, j = divmod(t, 10)
            if board[i, j] == 1:
                self.board_button_array[t].config(bg='black')
            else:
                self.board_button_array[t].config(bg='white')

    def update_pieces(self, sess_pieces):
        for t in range(3):
            if t >= len(sess_pieces):
                for f in self.piece_grid_array[t]:
                    f.config(bg="white")
            else:
                piece = pieces[sess_pieces[t]]
                p_size = np.shape(piece)
                for k in range(25):
                    i, j = divmod(k, 5)

                    if i < p_size[0] and j < p_size[1] and piece[i, j] == 1:
                        self.piece_grid_array[t][k].config(bg='black')
                    else:
                        self.piece_grid_array[t][k].config(bg='white')

        self.piece_button_array[self.piece_button_value].config(bg='lightgray')
        self.piece_button_value = -1



    def update_score(self, score):
        self.score_text.config(text="Score: %i" % score)


    def write_action(self, action: Action, reward, state_pieces, state_board):
        with open(self.save_file, 'a+') as f:
            p = [-1, -1, -1]
            for i in range(3):
                if i < len(state_pieces):
                    p[i] = state_pieces[i]
            f.write('%i|%i|%i|%i|%i|%i|%i|%s\n' % (action.p_id, action.pos.i, action.pos.j, reward, p[0], p[1], p[2],
                                        np.array_str(state_board.reshape((100)), max_line_width=500)))

    def show(self):
        self.root.mainloop()

    def stop(self):
        self.root.destroy()

ui = GameUI()

ui.show()
#time.sleep(30)
#ui.stop()
