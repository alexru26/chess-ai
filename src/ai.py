import math
import numpy as np
import chess
import hashlib
import json
from keras import models
import time

class AI:

    def __init__(self):
        self.model = models.load_model('../models/ai.keras')
        with open('dict.json', 'r') as f:
            self.int_to_move = json.load(f)

    def space_to_int(self, space):
        row_val = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        letter = chess.square_name(space)  # returns letter of space (ex. a3)
        return 8 - int(letter[1]), row_val[letter[0]]  # returns row, column

    def split_boards(self, board):
        """
        Splits board into 14 different boards
        :param board: current board
        :return: length 14 array of boards based on piece location and attack
        0-5 are white pieces (pawn, knight, bishop, rook, queen, king)
        6-11 is black pieces (pawn, knight, bishop, rook, queen, king)
        12 is white legal moves
        13 is black legal moves
        """
        res = np.zeros((14, 8, 8), dtype=np.int8)  # creates array of length 14 where each element is a board

        for piece in chess.PIECE_TYPES:
            for square in board.pieces(piece, chess.WHITE):
                # gets the location of where the square is and stores it into res
                index = np.unravel_index(square, (8,8))  # see https://stackoverflow.com/questions/48135736/what-is-an-intuitive-explanation-of-np-unravel-index
                res[piece - 1][7 - index[0]][index[1]] = 1  # 7-index[0] is row, index[1] is column
            for square in board.pieces(piece, chess.BLACK):
                # same idea but black pieces this time
                index = np.unravel_index(square, (8, 8))
                res[piece + 5][7 - index[0]][index[1]] = 1  # +5 because of res indexing

        turn = board.turn  # store current turn to use later

        # get legal moves for white
        board.turn = chess.WHITE
        for move in board.legal_moves:
            r, c = self.space_to_int(move.to_square)
            res[12][r][c] = 1

        # get legal moves for black
        board.turn = chess.BLACK
        for move in board.legal_moves:
            r, c = self.space_to_int(move.to_square)
            res[13][r][c] = 1

        board.turn = turn  # restore original turn

        return res

    def predict(self, board):
        data = self.split_boards(board).reshape(1, 14, 8, 8)
        predictions = self.model.predict(data, verbose=0)[0]
        legal_moves = list(board.legal_moves)
        legal_moves_uci = [move.uci() for move in legal_moves]
        sorted_indices = np.argsort(predictions)[::-1]
        for move_index in sorted_indices:
            move = self.int_to_move[str(move_index)]
            if move in legal_moves_uci:
                return move
        return None