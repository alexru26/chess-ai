import math
import numpy as np
import chess
import hashlib
from keras import models
import time

class AI:

    def __init__(self, color, depth=5, time_limit=5):
        self.model = models.load_model('../models/ai.keras')
        self.color = color
        self.depth = depth
        self.transposition_table = {}
        self.time_limit = time_limit
        self.explored = 0

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

    def static_eval(self, board):
        data = np.array([self.split_boards(board)])
        eval = self.model.predict(data, verbose=0)[0][0]
        eval = round(eval, 5)
        return eval

    def iterative_deepening(self, board, max_depth):
        """
        Iterative deepening search. Gradually increases the search depth.
        """
        best_move = None
        start_time = time.time()

        for depth in range(1, max_depth + 1):
            best_eval, best_move = self.minimax(board, depth, self.color == chess.WHITE, -math.inf, math.inf)
            # Break if time limit is exceeded
            if (time.time() - start_time) > self.time_limit:
                break

        return best_move

    def minimax(self, board, depth, maximizing, alpha, beta):
        """
        Minimax algorithm with alpha-beta pruning and transposition table.
        """
        # Hash the current board position to check if it's in the transposition table
        board_hash = self.zobrist_hash(board)
        if board_hash in self.transposition_table and self.transposition_table[board_hash]["depth"] >= depth:
            return self.transposition_table[board_hash]["value"], self.transposition_table[board_hash]["best_move"]

        if depth == 0 or board.is_game_over():
            return self.quiescence_search(board, alpha, beta), None  # Use quiescence search for leaf nodes

        legal_moves = self.order_moves(board)  # Order the moves
        best_move = None

        if maximizing:
            max_eval = -math.inf
            for move in legal_moves:
                self.explored += 1
                board.push(move)  # Make move
                eval, _ = self.minimax(board, depth - 1, False, alpha, beta)
                board.pop()  # Undo move

                if eval > max_eval:
                    max_eval = eval
                    best_move = move

                alpha = max(alpha, max_eval)
                if beta <= alpha:
                    break  # Alpha-beta pruning

            # Store the evaluation in the transposition table
            self.transposition_table[board_hash] = {"value": max_eval, "depth": depth, "best_move": best_move}
            return max_eval, best_move

        else:  # minimizing for black
            min_eval = math.inf
            for move in legal_moves:
                self.explored += 1
                board.push(move)  # Make move
                eval, _ = self.minimax(board, depth - 1, True, alpha, beta)
                board.pop()  # Undo move

                if eval < min_eval:
                    min_eval = eval
                    best_move = move

                beta = min(beta, min_eval)
                if beta <= alpha:
                    break  # Alpha-beta pruning

            # Store the evaluation in the transposition table
            self.transposition_table[board_hash] = {"value": min_eval, "depth": depth, "best_move": best_move}
            return min_eval, best_move

    def quiescence_search(self, board, alpha, beta):
        """
        Quiescence search evaluates quiet positions and continues searching for captures to avoid the horizon effect.
        """
        stand_pat = self.static_eval(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        legal_moves = list(board.legal_moves)

        # Only consider capture moves and promotions in quiescence search
        for move in legal_moves:
            if board.is_capture(move) or move.promotion:
                board.push(move)
                score = -self.quiescence_search(board, -beta, -alpha)
                board.pop()

                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score

        return alpha

    def order_moves(self, board):
        """
        Order moves to improve alpha-beta pruning by prioritizing certain types of moves.
        Prioritize:
        1. Captures
        2. Checks
        3. Promotions
        """
        legal_moves = list(board.legal_moves)

        def move_priority(move):
            # Higher value for captures, checks, and promotions
            score = 0
            if board.is_capture(move):
                score += 10  # Capture is high priority
            if board.gives_check(move):
                score += 5  # Check is also high priority
            if move.promotion:
                score += 20  # Promotion is very high priority
            return score

        # Sort moves by their priority in descending order
        ordered_moves = sorted(legal_moves, key=move_priority, reverse=True)
        return ordered_moves

    def zobrist_hash(self, board):
        """
        Generate a Zobrist hash for the current board position.
        Zobrist hashing is a technique for representing the game state as a unique hash.
        """
        return hashlib.sha256(board.fen().encode('utf-8')).hexdigest()

    def eval(self, board):
        self.explored = 0

        # printing
        # print('\nFinding best move...')

        # minimax initial call
        move = self.iterative_deepening(board, self.depth)

        # printing
        # print('- Boards explored', self.explored)

        return move