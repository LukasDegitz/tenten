import numpy as np
from utils import get_pieces, Action, little_gauss

class Session:

    board = None
    pieces = None
    score = None

    def __init__(self):
        self.board = np.zeros((10, 10), dtype=int)
        self.pieces = get_pieces()
        self.score = 0

    def print_state(self):
        print(self.score)
        print(self.board)
        for p in self.pieces:
            print(p)

    def take_action(self, action=Action):

        piece = self.pieces[action.pid]

        #check if selected piece is valid
        if not piece:
            return -1
        piece_shape = np.shape(piece)

        #check if selected piece fits to selected position
        if any(np.sum((piece_shape, Action.pos), axis=0) > 10):
            return -1
        if any(self.board[Action.b_i:Action.b_i + piece_shape[0], Action.b_j:Action.b_j + piece_shape[1]] + piece > 1):
            return -1

        #place piece on board
        self.board[Action.b_i:Action.b_i + piece_shape[0], Action.b_j:Action.b_j + piece_shape[1]] += piece

        #clear full rows/cols
        b_is, b_js = np.where(self.board.sum(axis=1) == 10)[0], np.where(self.board.sum(axis=0) == 10)[0]
        self.board[b_is, :] = 0
        self.board[:, b_js] = 0

        #score
        return little_gauss(len(b_is)+len(b_js)) + piece.sum()

    def piece_fits_pos(self, piece, pos):

        piece_shape = np.shape(piece)
        #dimension check
        if any(np.sum((piece_shape, pos), axis = 0) > 10):
            return False
        #position already occupied


        return True
