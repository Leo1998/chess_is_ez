
import chess
import chess.svg
import base64
import numpy as np
from game_state import GameState
from keras.models import load_model
from flask import Flask

model = load_model('models/net-1M-40E-v2.model')

#wtf keras???
model._make_predict_function()

def stateToSvg(state):
    return base64.b64encode(chess.svg.board(board=state.board).encode('utf-8')).decode('utf-8')

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

app = Flask(__name__)

@app.route("/")
def root():
    return "Test my ass"

@app.route("/selfplay")
def selfplay():
    s = GameState()

    ret = '<html><head>'
    
    while not s.board.is_game_over():
        ai_move(s)
        ret += '<img width=600 height=600 src="data:image/svg+xml;base64,{}"></img><br/>'.format(stateToSvg(s))
        #print(s.board.result())

    return ret

if __name__ == '__main__':
    app.run()