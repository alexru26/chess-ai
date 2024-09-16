import copy, math, random

from const import *
from piece import *
import numpy as np
from keras import models

class AI:

    def __init__(self, depth=1):
        self.depth = depth
        self.model = models.load_model('../models/ai.keras')
        self.color = 'black'
        self.explored = 0

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

        piece_types = {0: 'pawn', 1: 'knight', 2: 'bishop', 3: 'rook', 4: 'queen', 5: 'king'}

        for piece_type in piece_types:
            # gets location of white piece and stores into res
            for row in range(ROWS):
                for col in range(COLS):
                    if board.squares[row][col].has_piece():
                        piece = board.squares[row][col].piece
                        if piece.name == piece_types[piece_type] and piece.color == 'white':
                            res[piece_type][row][col] = 1

            # gets location of black piece and stores into res
            for row in range(ROWS):
                for col in range(COLS):
                    if board.squares[row][col].has_piece():
                        piece = board.squares[row][col].piece
                        if piece.name == piece_types[piece_type] and piece.color == 'black':
                            res[piece_type+6][row][col] = 1

        # get legal moves for white
        white_moves = self.get_moves(board, 'white')
        for move in white_moves:
            r = move.final.row
            c = move.final.col
            res[12][r][c] = 1

        # get legal moves for black
        black_moves = self.get_moves(board, 'black')
        for move in black_moves:
            r = move.final.row
            c = move.final.col
            res[13][r][c] = 1

        return res

    def static_eval(self, board):
        data = np.array([self.split_boards(board)])

        eval = self.model.predict(data, verbose=0)[0][0]
        
        eval = round(eval, 5)
        return eval

    def get_moves(self, board, color):
        moves = []
        for row in range(ROWS):
            for col in range(COLS):
                square = board.squares[row][col]
                if square.has_team_piece(color):
                    board.calc_moves(square.piece, square.row, square.col)
                    moves += square.piece.moves
        
        return moves

    def order_moves(self, moves, board):
        """Order moves to prioritize captures and checks."""
        ordered_moves = []
        for move in moves:
            if board.squares[move.initial.row][move.initial.col].piece is None: continue
            target_square = board.squares[move.final.row][move.final.col]
            if target_square.has_piece():
                ordered_moves.insert(0, move)
            else:
                ordered_moves.append(move)  # Normal moves

        return ordered_moves

    def minimax(self, board, depth, maximizing, alpha, beta):
        if depth == 0:
            return self.static_eval(board), None # eval, move
        
        # white
        if maximizing:
            max_eval = -math.inf
            moves = self.order_moves(self.get_moves(board, 'white'), board)
            for move in moves:
                self.explored += 1
                piece = board.squares[move.initial.row][move.initial.col].piece
                temp_board = copy.deepcopy(board)
                temp_board.move(piece, move)
                piece.moved = False
                eval = self.minimax(temp_board, depth-1, False, alpha, beta)[0] # eval, mov
                if eval > max_eval:
                    max_eval = eval
                    best_move = move

                alpha = max(alpha, max_eval)
                if beta <= alpha: break

            if not best_move:
                best_move = moves[0]

            return max_eval, best_move # eval, move
        
        # black
        elif not maximizing:
            min_eval = math.inf
            moves = self.order_moves(self.get_moves(board, 'black'), board)
            for move in moves:
                self.explored += 1
                piece = board.squares[move.initial.row][move.initial.col].piece
                temp_board = copy.deepcopy(board)
                temp_board.move(piece, move)
                piece.moved = False
                eval = self.minimax(temp_board, depth-1, True, alpha, beta)[0] # eval, move
                if eval < min_eval:
                    min_eval = eval
                    best_move = move

                beta = min(beta, min_eval)
                if beta <= alpha: break
            
            if not best_move:
                idx = random.randrange(0, len(moves))
                best_move = moves[idx]

            return min_eval, best_move # eval, move

    def eval(self, main_board):
        self.explored = 0

        # printing
        print('\nFinding best move...')

        # minimax initial call
        eval, move = self.minimax(main_board, self.depth, False, -math.inf, math.inf)

        # printing
        print('- Boards explored', self.explored)

        return move