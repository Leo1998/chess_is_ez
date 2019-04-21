import chess
import numpy as np

class GameState(object):
    def __init__(self, board=None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board
    
    # the serialize method is inspired by George Hotz' twitchchess. Maybe going to write better function....
    def serialize(self):
        assert self.board.is_valid()

        #every piece is coded in 2^4 bits
        binary_state = np.zeros(64, np.uint8)

        for i in range(64):
            piece = self.board.piece_at(i)
            if piece is not None:
                binary_state[i] = {"P": 1, "N": 2, "B": 3, "R": 4, "Q": 5, "K": 6, \
                     "p": 9, "n":10, "b":11, "r":12, "q":13, "k": 14}[piece.symbol()]

        if self.board.has_queenside_castling_rights(chess.WHITE):
            assert binary_state[0] == 4
            binary_state[0] = 7
        if self.board.has_kingside_castling_rights(chess.WHITE):
            assert binary_state[7] == 4
            binary_state[7] = 7
        if self.board.has_queenside_castling_rights(chess.BLACK):
            assert binary_state[56] == 8+4
            binary_state[56] = 8+7
        if self.board.has_kingside_castling_rights(chess.BLACK):
            assert binary_state[63] == 8+4
            binary_state[63] = 8+7

        # en passant
        if self.board.ep_square is not None:
            assert binary_state[self.board.ep_square] == 0
            binary_state[self.board.ep_square] = 8

        #make 2D
        binary_state = binary_state.reshape(8, 8)

        state = np.zeros((5,8,8), np.uint8)

        state[0] = (binary_state>>3)&1
        state[1] = (binary_state>>2)&1
        state[2] = (binary_state>>1)&1
        state[3] = (binary_state>>0)&1

        # 4th column is who's turn it is
        state[4] = (self.board.turn*1.0)

        return state