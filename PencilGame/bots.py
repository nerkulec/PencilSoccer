from PencilGame.pencilGame import Game
import numpy as np
from keras.models import load_model
import random


class Bot:
    def __init__(self, game=None):
        self.game = game

    def get_move(self):
        pass


class SearchBot(Bot):  # TODO: Real priority queued search
    def __init__(self, game=None, max_depth=1):
        super().__init__(game=game)
        self.max_depth = max_depth
        self.best_move = None

    def get_score(self):  # Close to P1's goal -> negative, middle -> 0, close to P2's goal -> positive
                          # P1 won -> 1000000000, P2 won -> -1000000000
        if self.game.winner is None:
            return self.game.ball[1] - (self.game.height - 1) // 2
        else:
            return 1000000000 if self.game.winner is "P1" else -1000000000

    def search(self, depth, set_best_move=False):
        """Returns [lowest] height you can achieve in *depth* moves if turn == P1 else [highest]"""
        if depth <= 0:
            return self.get_score()

        game_scores = []
        for dir_num in range(len(self.game.move_vectors)):
            map, score, winner, overwriting = self.game.step(move_num=dir_num)
            if winner:
                game_scores.append(score)
            else:
                game_scores.append(self.search(depth - 1))
            self.game.undo_last_step(overwriting)
        if self.game.turn is "P1":
            if set_best_move:
                self.best_move = np.argmax(game_scores)
            return max(game_scores)
        else:
            if set_best_move:
                self.best_move = np.argmin(game_scores)
            return min(game_scores)

    def get_move(self):  # TODO: if choice is ambiguous choose randomly
        self.best_move = None
        self.search(self.max_depth, set_best_move=True)
        return self.best_move


class NNBot(Bot):  # has to flip the board himself if player is P2
    def __init__(self, game=None, version=None, prevent_blunder=False):
        super().__init__(game)
        self.model = load_model('saves/pencilbot_{}.h5'.format(version))
        self.prevent_blunder = prevent_blunder

    def get_move(self):  # TODO: prevent blunder
        values = []
        for move_num in range(8):
            _, _, winner, overwriting = self.game.step(move_num)
            values.append(self.model.predict(np.array([self.game.get_env()]))[0][0])
            self.game.undo_last_step(overwriting)
        print(np.array(values))
        return np.argmax(values) if self.game.turn is "P1" else np.argmin(values)


class RandomBot(Bot):
    def __init__(self, game=None, prevent_blunder=False):
        super().__init__(game)
        self.prevent_blunder = prevent_blunder

    def get_move(self):
        if self.prevent_blunder:
            possibilities = [x for x in range(8)]
            while possibilities:
                move_num = random.choice(possibilities)
                possibilities.remove(move_num)
                try:
                    self.game.test(move_num)
                    return move_num
                except GameOver as e:
                    if e.winner is not self.game.turn:
                        continue
            return random.randrange(8)
        else:
            return random.randrange(8)
