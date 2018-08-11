from PencilGame.pencilGame import GameOver, Game
import numpy as np
import random


class Bot:
    def __init__(self, game=None):
        self.true_game = game

    def update(self, game):
        self.true_game = game

    def get_move(self):
        pass


class SearchBot(Bot):  # TODO: Real priority queued search with distance from both goals metric
    def __init__(self, game=None, max_depth=1):
        super().__init__(game=game)
        self.max_depth = max_depth
        self.best_move = None

    def update(self, game):
        super().update(game)
        self.best_move = None

    def search(self, game, depth, set_best_move=False):
        """Returns lowest height you can achieve in *depth* moves if turn == P1 else ^highest"""
        if depth <= 0:
            return game.ball[1]

        game_scores = []
        for dir_num in range(len(self.true_game.moves)):
            test_game = game.copy()
            try:
                test_game.move(move_num=dir_num)
                game_scores.append(self.search(test_game, depth-1))  # Does not happen if GameOver
            except GameOver as e:
                if e.winner is "P1":
                    game_scores.append(1000)
                if e.winner is "P2":
                    game_scores.append(-1000)
            # print("TEST")
            # [print(line) for line in test_game.render_map()]
        if game.turn is "P1":
            if set_best_move:
                self.best_move = np.argmax(game_scores)
            return max(game_scores)
        else:
            if set_best_move:
                self.best_move = np.argmin(game_scores)
            return min(game_scores)

    def get_move(self):
        self.search(self.true_game, self.max_depth, set_best_move=True)
        return self.best_move


class NNBot(Bot):  # has to flip the board himself if player is P2
    def __init__(self, game=None, model=None, prevent_blunder=False):
        super().__init__(game)
        self.model = model
        self.prevent_blunder = prevent_blunder

    def get_move(self):
        env = self.true_game.get_env()
        if self.true_game.turn is "P2":
            env = np.rot90(env, k=2, axes=(1, 2))
        votes = self.model.predict(np.array([env]))
        if self.true_game.turn is "P2":
            votes = np.roll(votes, 4)
        if self.prevent_blunder:
            for (move_num, vote) in sorted(enumerate(votes), key=lambda x: x[1], reverse=True):
                print(move_num)
                try:
                    self.true_game.test(move_num)
                    return move_num
                except GameOver as e:
                    if e.winner is self.true_game.turn:
                        return move_num
                    else:
                        continue
            return np.argmax(votes)
        else:
            return np.argmax(votes)


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
                    self.true_game.test(move_num)
                    return move_num
                except GameOver as e:
                    if e.winner is not self.true_game.turn:
                        continue
            return random.randrange(8)
        else:
            return random.randrange(8)


if __name__ == "__main__":
    game = Game()
    brute = SearchBot(game=game, max_depth=2)
    while True:
        [print(line) for line in game.render_map()]
        try:
            if game.turn is "P1":
                move = game.move_num(input("({},{}) {}:".format(game.ball[0], game.ball[1], game.turn)))
            else:
                brute.update(game)
                move = brute.get_move()
            game.move(move_num=move)
        except KeyError as e:
            print(e)
        except GameOver as e:
            print("Winner is {}".format(e.winner))
            print(e.message)
            break
