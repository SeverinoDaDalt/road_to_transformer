# How to work with chess data

## chess python module
Chess module in python is super useful to work with .pgn files.
For example, to load first game from a .pgn file, you can:
```python
import chess.pgn

pgn_file = open("file.pgn")
first_game = chess.pgn.read_game(pgn_file)
```

You can also get the list of moves in the game by running:
```python
test = []
board = first_game.board()
for move in first_game.mainline_moves():
    test.append(board.san(move))
    board.push(move)
```
