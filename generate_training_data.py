import os
import chess.pgn
import numpy as np
from game_state import GameState

def get_dataset(num_samples=None):
  X,Y = [], []
  games_parsed = 0
  values = {'1/2-1/2':0, '0-1':-1, '1-0':1}

  for fn in os.listdir("data"):
    pgn = open(os.path.join("data", fn), encoding="ISO-8859-1")
    while 1:
      game = chess.pgn.read_game(pgn)
      if game is None:
        break
      res = game.headers['Result']
      if res not in values:
        continue
      value = values[res]
      board = game.board()
      for i, move in enumerate(game.mainline_moves()):
        board.push(move)
        ser = GameState(board).serialize()
        X.append(ser)
        Y.append(value)
      print("Parsed {} games, got {} moves".format(games_parsed, len(X)))
      if num_samples is not None and len(X) > num_samples:
        return X,Y
      games_parsed += 1
  X = np.array(X)
  Y = np.array(Y)
  return X,Y

if __name__ == "__main__":
  #X,Y = get_dataset(10000)
  #np.save("parsed_data/X_10K.npy", X)
  #np.save("parsed_data/Y_10K.npy", Y)

  X,Y = get_dataset(1000000)
  np.save("parsed_data/X_1M.npy", X)
  np.save("parsed_data/Y_1M.npy", Y)

  #X,Y = get_dataset(10000000)
  #np.save("parsed_data/X_10M.npy", X)
  #np.save("parsed_data/Y_10M.npy", Y)
