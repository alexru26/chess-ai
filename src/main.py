import chess
from chessboard import display
from ai import AI
import random

# Initialize the chess board
board = chess.Board()
display_board = display.start()

# Randomly decide if the player is white or black
player_is_white = random.choice([True, False])

ai = AI()

def player_move():
    move_made = False
    while not move_made:
        try:
            user_move = input("Enter your move in UCI format (e.g., e2e4): ")
            move = chess.Move.from_uci(user_move)
            if move in board.legal_moves:
                board.push(move)
                move_made = True
            else:
                print("Illegal move. Try again.")
        except ValueError:
            print("Invalid input. Please use UCI format (e.g., e2e4).")

def computer_move():
    move = ai.predict(board)
    board.push_uci(move)

def main():
    # Inform the player of their color
    if player_is_white:
        print("You are playing as White!")
    else:
        print("You are playing as Black!")

    while not board.is_game_over():
        display.check_for_quit()
        if (board.turn == chess.WHITE and player_is_white) or (board.turn == chess.BLACK and not player_is_white):
            player_move()
            display.update(board.fen(), display_board)
        else:
            computer_move()
            display.update(board.fen(), display_board)

    print("Game Over!")
    if board.is_checkmate():
        print("Checkmate!")
    elif board.is_stalemate():
        print("Stalemate!")
    elif board.is_insufficient_material():
        print("Draw by insufficient material.")
    elif board.is_seventyfive_moves():
        print("Draw by seventy-five moves rule.")
    elif board.is_fivefold_repetition():
        print("Draw by fivefold repetition.")
    elif board.is_variant_draw():
        print("Draw by variant rule.")

if __name__ == "__main__":
    main()