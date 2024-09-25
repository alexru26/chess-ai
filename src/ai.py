import json
import random

import chess
import numpy as np
from keras import models


class AI:
    """AI class that gives optimal move"""

    def __init__(self):
        """
        AI class initializer
        Loads ai model and dictionary
        """
        self.model = models.load_model('../models/ai.keras')
        with open('dict.json', 'r') as f:
            self.int_to_move = json.load(f)

    def split_boards(self, board):
        """
        Splits board into 12 different boards
        Taken from here: https://github.com/Skripkon/chess-engine
        :param board: current board
        :return: 12 8x8 boards based on piece location and attack
        0-5 are white pieces (pawn, knight, bishop, rook, queen, king)
        6-11 is black pieces (pawn, knight, bishop, rook, queen, king)
        """
        matrix = np.zeros((8, 8, 12))
        piece_map = board.piece_map()
        for square, piece in piece_map.items():
            row, col = divmod(square, 8)
            piece_type = piece.piece_type - 1
            piece_color = 0 if piece.color else 6
            matrix[row, col, piece_type + piece_color] = 1
        return matrix

    def predict(self, board):
        """
        Uses model to predict best move and makes sure it is legal
        Taken from https://github.com/Skripkon/chess-engine
        """
        data = self.split_boards(board).reshape(1, 8, 8, 12) # transforms board
        predictions = self.model.predict(data, verbose=0)[0] # makes prediction
        legal_moves = list(board.legal_moves)
        legal_moves_uci = [move.uci() for move in legal_moves] # convert legal moves to uci
        sorted_indices = np.argsort(predictions)[::-1]
        for move_index in sorted_indices:
            move = self.int_to_move[str(move_index)] # convert int to move
            if move in legal_moves_uci: # if is legal move
                return move
        return random.choice(legal_moves_uci)
