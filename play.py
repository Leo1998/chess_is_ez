
import chess
import chess.svg
import base64
import numpy as np
from game_state import GameState
from keras.models import load_model
from flask import Flask

model = load_model('models/net-100E-10K-eval.model')

#wtf keras???
model._make_predict_function()

def stateToSvg(state):
    return chess.svg.board(board=state.board).encode('utf-8')

def eval_move(state):
    tensor = state.serialize()
    res = model.predict(tensor[None])
    return res[0][0]

def ai_move(state):
    sorted_moves = []
    for e in state.edges():
        state.board.push(e)
        value = eval_move(state)
        sorted_moves.append((value, e))
        state.board.pop()
    sorted_moves = sorted(sorted_moves, key=lambda x: x[0], reverse=state.board.turn)

    if len(sorted_moves) == 0:
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
            traceback.print_exc()

        ai_move(s)
        self.chessView.load(stateToSvg(s))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())