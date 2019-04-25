

import chess

def stateToSvg(state):
    return chess.svg.board(board=state.board).encode('utf-8')

from flask import Flask

app = Flask(__name__)

@app.route("/")
def root():
    return "Test my ass"

if __name__ == '__main__':
    app.run()