# Machine Learning Project 01: Chess AI
This is an AI that plays chess alright. 

## How to run
Make sure you have the necessary packages.

```
pip install pandas
pip install tensorflow
pip install keras
pip install chess
pip install chess-board
```

Run main.py to play against the AI. 

## Resources Used
- ChatGPT
- https://github.com/Skripkon/chess-engine
- https://www.youtube.com/watch?v=ffzvhe97J4Q
- Data from https://database.nikonoel.fr/
  - Data is not included in the repository because it is 3.2 GB

## How it works
- Model is trained on 27728 games and 2276758 different positions
- Sequential model with two Conv2D layers, a Flatten layer, and two Dense layers
- The model is fed an 8x8x12 array that represents where the 6 white and 6 black pieces are located
- The model returns an integer, which represents a move
- A dictionary is used to convert from moves (ex. e2e4) to integers (ex. 1833) and vice versa

<img width="598" alt="Screenshot 2024-09-24 at 12 59 32 PM" src="https://github.com/user-attachments/assets/498bce57-5e56-4911-bfd9-9ef051b0c706">

*Image 1. AI architecture.*

<img width="529" alt="Screenshot 2024-09-24 at 12 59 59 PM" src="https://github.com/user-attachments/assets/0c006625-2c78-4b52-b66e-c3fd58906798">

*Image 2. Loss and accuracy graph of model after training. The model visibly overfitted towards the end.*

## Reasoning and Issues
- First attempt: Eval predicter + Minimax
  - Poor Eval predictions
    - Requires a lot of data and therefore time to be accurate
    - Predicter losses could compound in minimax and make AI unreliable
  - Minimax was slow
    - Even at only depth 3, it took a few minutes to make a single move
    - For comparison, Stockfish engine on chess.com runs at 14-16 depth
- Second attempt: Magnus Carlsen predicter
  - Poor predictions despite being 60% accurate
    -  Lack of data
      - Good at the start but blunders a lot during midgame and endgame
    -  Magnus likes to troll
- Final attempt: Lichess database predicter
  -  More well-rounded data
  -  Players are worse than Magnus, but they are still good (2000+ on lichess)
  -  Also gets worse throughout the game but not as bad as Magnus predicter
  -  Resorts to a random legal move if predicted move is not legal

## Further Improvements
- Load more files and more games
  - AI would be trained on more scenarios and thus would be more prepared
    - A very small amount of the 3.2 GB has been used to train the model
- Make AI more complicated
  - Change architecture of AI
    - It's possible the AI is limited by its basic architecture
