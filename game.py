import numpy as np
from utils import get_pieces, Action, Position, State, little_gauss, base_position_mask, pieces, transformed_pieces

class Session:

    board = None
    pieces = None
    score = None
    lost = None

    position_mask = None
    piece_vector = None

    def __init__(self):
        self.board = np.zeros((10, 10))
        self.pieces = get_pieces()
        self.score = 0
        self.lost = False
        self.position_mask = base_position_mask.copy()
        self._update_piece_vector()

    def print_state(self):
        print(self.score)
        print(self.board)
        for p in self.pieces:
            print(p)
            print(pieces[p])

    def get_state(self):
        return State(self.piece_vector, self.position_mask)

    def take_action(self, action: Action):

        piece_id, target_position = action.p_id, action.pos

        # check if the action is valid
        if piece_id not in self.pieces or piece_id > len(pieces):
            return -1

        piece = pieces[piece_id]
        piece_shape = np.shape(piece)

        # check if piece fits into the desired position
        if self.position_mask[piece_id, target_position.i, target_position.j] == 0:
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
        self._update_position_mask()

        if np.multiply(self.piece_vector.reshape((19, 1, 1)), self.position_mask).sum() == 0:
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

    def _update_position_mask(self):

        #invert and transform board
        board_inv = np.logical_not(self.board).astype(int)
        board_trans = np.fft.fft2(board_inv)

        # convolute all possible pieces with the inverted board to get the mask of all possible
        # positions. Make use of the fft convolution trick.
        for piece_id, (piece, piece_trans) in enumerate(zip(pieces, transformed_pieces)):

            self.position_mask[piece_id] = (np.real(np.fft.ifft2(board_trans*np.conj(piece_trans))) >= piece.sum()).astype(int)

        self.position_mask = np.multiply(self.position_mask, base_position_mask)

    def _remove_piece(self, pid):

        if pid not in self.pieces:
            return

        self.pieces.remove(pid)

        if not self.pieces:
            self.pieces = get_pieces()

        self._update_piece_vector()

    def _update_piece_vector(self):
        self.piece_vector = np.zeros((19,))
        for piece in self.pieces:
            self.piece_vector[piece] = 1
