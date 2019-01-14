import random
import numpy as np
from collections import namedtuple
from game2048.expectimax import board_to_move
from grid_ohe import grid_ohe

Guide = namedtuple('Guide', ('state', 'action'))


class Guides:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Guide(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def ready(self, batch_size):
        return len(self.memory) >= batch_size

    def __len__(self):
        return len(self.memory)


class ModelWrapper:
    def __init__(self, model, capacity):
        self.model = model
        self.memory = Guides(capacity)
        self.training_step = 0

    def predict(self, board):
        return self.model.predict(np.expand_dims(board, axis=0))

    def move(self, game):
        ohe_board = grid_ohe(game.board)
        suggest = board_to_move(game.board)
        direction = self.predict(ohe_board).argmax()
        game.move(direction)
        self.memory.push(ohe_board, suggest)

    def train(self, batch):
        if self.memory.ready(batch):
            guides = self.memory.sample(batch)
            X = []
            Y = []
            for guide in guides:
                X.append(guide.state)
                ohe_action = [0] * 4
                ohe_action[guide.action] = 1
                Y.append(ohe_action)
            loss, acc = self.model.train_on_batch(np.array(X), np.array(Y))
            self.training_step += 1
