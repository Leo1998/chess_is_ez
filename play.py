
import chess
import chess.svg
import base64
import math
import numpy as np
from game_state import GameState
from keras.models import load_model
from flask import Flask

def signum(x):
    if x >= 0:
        return 1
    else:
        return -1

model = load_model('models/net-1M-60E-v2.model')

#wtf keras???
model._make_predict_function()

def stateToSvg(state):
    return chess.svg.board(board=state.board).encode('utf-8')

def eval_move(state):
    tensor = state.serialize()
    res = model.predict(tensor[None])
    return res[0][0]

def explore_moves(state, depth, turn_pov):
    moves = []
    edges = state.edges()

    for e in edges:
        state.board.push(e)

        if not state.board.is_valid():
            state.board.pop()
            continue

        value = eval_move(state)

        #opponent move
        thres = 0.4
        if turn_pov == state.board.turn:
            turn_signum = signum(state.board.turn * 1.0 - 0.5)
            if signum(value) != turn_signum or abs(value) <= thres:
                continue



        if depth > 1:
            next_moves = explore_moves(state, depth-1, turn_pov)

            if len(next_moves) == 0:
                continue

            best_move = next_moves[0]
            moves.append((best_move[0], e))
        else:
            moves.append((value, e))

        state.board.pop()

    return sorted(moves, key=lambda x: x[0], reverse=turn_pov)

def ai_move(state):
    sorted_moves = explore_moves(state, 1, state.board.turn)

    if len(sorted_moves) == 0:
        print("AI can't move...")
        return
    
    print("Top 3:")
    for i,m in enumerate(sorted_moves[0:3]):
        print("    Score: ", i, " Move: ", m)

    state.board.push(sorted_moves[0][1])

s = GameState()

from PyQt5 import QtSvg
from PyQt5.QtWidgets import QApplication, QMainWindow, QLineEdit, QPushButton
import sys


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('chezzzz')
        self.setGeometry(100, 100, 600, 650)

        self.chessView = QtSvg.QSvgWidget(self)
        self.chessView.resize(600, 600)
        self.chessView.load(stateToSvg(s))

        self.textbox = QLineEdit(self)
        self.textbox.move(0, 600)
        self.textbox.resize(400, 50)
        self.textbox.editingFinished.connect(self.do_move)

        self.button = QPushButton("Move", self)
        self.button.move(400, 600)
        self.button.resize(200, 50)
        self.button.clicked.connect(self.do_move)

        self.show()

    def do_move(self):
        move = self.textbox.text()
        self.textbox.setText("")

        try:
            s.board.push(s.board.parse_uci(move))
            self.chessView.load(stateToSvg(s))
        except Exception:
            print("Invalid move: {}".format(move))
            return

        if s.board.is_game_over():
            return

        ai_move(s)
        self.chessView.load(stateToSvg(s))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())