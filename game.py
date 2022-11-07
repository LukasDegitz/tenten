import numpy as np
from utils import get_pieces, Action, Position, little_gauss, base_position_mask, pieces

class Session:

    board = None
    pieces = None
    score = None
    lost = None

    position_mask = None
    piece_vector = None

    def __init__(self):
        self.board = np.zeros((10, 10), dtype=int)
        self.pieces = get_pieces()
        self.score = 0
        self.lost = False
        self.position_mask = base_position_mask.copy()
        self._update_piece_vector()

    def print_state(self):
        print(self.score)
        print(self.board)
        for p in self.pieces:
            print(pieces[p])

    def take_action(self, action=Action):

        if action.p_id not in self.pieces or action.p_id > len(pieces):
            return -1
        piece = pieces[action.p_id]

        fits = self.piece_fits_position(piece, action.pos, update=True)
        if not fits:
            return -1

        #else peice fitted and board was updated
        step_score = self.clear_rows() + piece.sum()

        #remove used piece
        self._remove_piece(action.p_id)

        #update state
        self._update_position_mask()

        if np.multiply(self.piece_vector.reshape((19, 1, 1)), self.position_mask).sum() == 0:
            #game over - no more possible moves
            self.score += step_score
            return -1000

        return step_score


    def piece_fits_position(self, piece, position: Position, update=False):

        piece_shape = np.shape(piece)
        # check if selected piece fits to selected position
        if any(np.sum((piece_shape, position), axis=0) > 10):
            return False

        if any(self.board[position.i:position.i + piece_shape[0], position.j:position.j + piece_shape[1]] + piece > 1):
            return False

        # place piece on board
        if update:
            self.board[position.i:position.i + piece_shape[0], position.j:position.j + piece_shape[1]] += piece

        return True

    def clear_rows(self):

        #clear full rows/cols
        b_is, b_js = np.where(self.board.sum(axis=1) == 10)[0], np.where(self.board.sum(axis=0) == 10)[0]
        self.board[b_is, :] = 0
        self.board[:, b_js] = 0

        #score
        return little_gauss(len(b_is)+len(b_js))

    def _update_position_mask(self):

        board_inv = self._invert_board()
        self.position_mask = np.multiply(board_inv, base_position_mask)

        for piece, position_matrix in zip(pieces, self.position_mask):
            possible_pos = np.where(position_matrix == 1)
            possible_pos_is, possible_pos_js = possible_pos[0], possible_pos[1]

            for possible_pos in zip(possible_pos_is, possible_pos_js):
                if not self.piece_fits_position(piece, possible_pos, update=False):
                    position_matrix[possible_pos] = 0

    def _remove_piece(self, pid):

        if pid not in self.pieces:
            return

        self.pieces.remove(pid)

        if not self.pieces:
            self.pieces = get_pieces()

        self._update_piece_vector()

    def _invert_board(self):
        return np.logical_not(self.board).astype(int)

    def _update_piece_vector(self):
        self.piece_vector = np.zeros((19,))
        for piece in self.pieces:
            self.piece_vector[piece] = 1
