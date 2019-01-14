from game2048.agents import Agent
from keras.models import load_model
from grid_ohe import grid_ohe
import numpy as np

my_model = load_model("model.h5")


class MyAgent(Agent):
    def step(self):
        game_board = np.expand_dims(grid_ohe(self.game.board),axis=0)
        direction = int(my_model.predict(game_board).argmax())
        return direction
