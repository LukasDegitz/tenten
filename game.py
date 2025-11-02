import numpy as np
from utils import get_pieces, Action, State, little_gauss, position_mask, pieces, transform_state
import torch

class Session:

    board = None
    pieces = None
    score = None
    lost = None

    mask = None
    piece_vector = None
    piece_mask = None

    def __init__(self):
        self.board = np.zeros((10, 10))
        self.pieces = get_pieces()
        self.score = 0
        self.lost = False
        self.mask = np.zeros((19, 10, 10))
        self._update_mask()
        self._update_piece_vector()

    def print_state(self):
        print(self.score)
        print(self.board)
        for p in self.pieces:
            print(p)
            print(pieces[p])

    def state_str(self):
        state_str = ''
        state_str += 'score: %i\n'%self.score
        state_str += np.array2string(self.board)
        state_str += '\n'
        for p in self.pieces:
            state_str += np.array2string(pieces[p])
            state_str += '\n'

        return state_str

    def get_state(self):
        return State(self.board.copy(), self.pieces.copy())

    def get_mask(self):
        return self.mask.copy()

    def get_transformed_state(self):
        return transform_state(self.get_state())

    def take_action(self, action: Action):

        piece_id, target_position = action.p_id, action.pos

        # check if the action is valid
        if piece_id not in self.pieces or piece_id > len(pieces):
            print('invalid piece')
            return -1

        piece = pieces[piece_id]
        piece_shape = np.shape(piece)

        # check if piece fits into the desired position
        if self.mask[piece_id, target_position.i, target_position.j] == 0:
            print('invalid position')
            print(self.mask[piece_id])
            print(action)
            self.print_state()
            return -1

        # update the board
        self.board[target_position.i:target_position.i + piece_shape[0],
                   target_position.j:target_position.j + piece_shape[1]] += piece

        #else peice fitted and board was updated
        step_score = self.clear_rows() + piece.sum()
        self.score += step_score

        #remove used piece
        self._remove_piece(action.p_id)

        #update state
        self._update_mask()

        if np.multiply(self.piece_vector.reshape((19, 1, 1)), self.mask).sum() == 0:
            #game over - no more possible moves
            self.lost = True
            return -1000

        return step_score


    def clear_rows(self):

        #clear full rows/cols
        b_is, b_js = np.where(self.board.sum(axis=1) == 10)[0], np.where(self.board.sum(axis=0) == 10)[0]
        self.board[b_is, :] = 0
        self.board[:, b_js] = 0

        #score
        return little_gauss(len(b_is)+len(b_js)) * 10

    def _update_mask(self):

        self.mask = position_mask(self.board.copy())

    def _remove_piece(self, pid):

        if pid not in self.pieces:
            return

        self.pieces.remove(pid)

        if not self.pieces:
            self.pieces = get_pieces()

        self._update_piece_vector()

    def _update_piece_vector(self):
        self.piece_vector = np.zeros((19,))
        self.piece_mask = np.zeros((19,))
        for piece in self.pieces:
            self.piece_vector[piece] += 1
            self.piece_mask[piece] = 1
