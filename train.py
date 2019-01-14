import numpy as np
import os
from guide import ModelWrapper
from model import build
from keras.models import load_model
from game2048.game import Game



if os.path.exists("model.h5"):
    my_model = load_model("model.h5")
else:
    my_model=build()

mw = ModelWrapper(my_model, 2**15)
BATCH_SIZE = 1024

times = 0
while True:
    game = Game()
    while not game.end:
        mw.move(game)

    mw.train(BATCH_SIZE)
    if times % 500 ==0:
        my_model.save("model.h5")


