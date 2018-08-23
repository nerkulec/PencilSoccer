import numpy as np


class Game:
    def __init__(self, width=7, height=8, lite=False):
        self.width = width
        self.height = height
        self.bot_P1 = None
        self.bot_P2 = None
        self.lite = lite
        self.move_vectors = [[0, -1], [1, -1], [1, 0], [1, 1],
                             [0, 1], [-1, 1], [-1, 0], [-1, -1]]  # UP-> 0, then clockwise
        self.map = self.gen_map()
        self.ball = [self.width // 2, (self.height - 1) // 2]
        self.turn = "P1"
        self.last_turn = "P2"
        self.P1_goal = [[self.width//2-1, 0],
                        [self.width//2, 0],
                        [self.width//2+1, 0]]  # TOP
        self.P2_goal = [[self.width//2-1, self.height-1],
                        [self.width//2, self.height-1],
                        [self.width//2+1, self.height-1]]  # BOTTOM
        self.move_history = []  # Format: [x, y, move_num, player]
        self.winner = None

    def set_bot_P1(self, bot):
        self.bot_P1 = bot

    def set_bot_P2(self, bot):
        self.bot_P2 = bot

    def copy(self):
        game_copy = Game(width=self.width, height=self.height, lite=self.lite)
        for move in self.move_history:
            game_copy.step(move_num=move[2])
        return game_copy

    def gen_map(self):
        full_map = np.zeros(shape=(self.width, self.height, len(self.move_vectors)))
        full_map[:, 0, :3] = 1
        full_map[:, 0, -2:] = 1
        full_map[:, self.height-1, 2:7] = 1
        full_map[self.width-1, :, 0:5] = 1
        full_map[0, :, 4:] = 1
        full_map[0, :, 0] = 1
        return full_map

    @staticmethod
    def move_num(move_char):
        return {"u": 0, "ur": 1, "ru": 1, "r": 2, "dr": 3, "rd": 3,
                "d": 4, "dl": 5, "ld": 5, "l": 6, "ul": 7, "lu": 7}[move_char]

    def _place_edge(self, pos, move_num):  # TODO: lite
        self.map[pos[0], pos[1], move_num] = 1
        new_pos = [pos[0] + self.move_vectors[move_num][0], pos[1] + self.move_vectors[move_num][1]]
        self.map[new_pos[0], new_pos[1], (move_num + 4) % 8] = 1

    def _erase_edge(self, pos, move_num):
        self.map[pos[0], pos[1], move_num] = 0
        new_pos = [pos[0] + self.move_vectors[move_num][0], pos[1] + self.move_vectors[move_num][1]]
        self.map[new_pos[0], new_pos[1], (move_num + 4) % 8] = 0

    def _is_edge(self, pos, move_num):  # TODO: lite
        return self.map[pos[0], pos[1], move_num] == 1

    def _is_escapable(self, pos, except_move_num=None):
        for move_num in [x for x in range(len(self.move_vectors)) if x is not except_move_num]:
            if self.map[pos[0], pos[1], move_num] == 0:
                return True
        return False

    def _is_in_graph(self, pos, last_move_num=None):
        """ Returns True if edge is in graph. If last_move_num is specified ignores last move """
        for move_num in range(len(self.move_vectors)):
            if last_move_num is None or move_num != (last_move_num + 4) % 8:
                if self._is_edge(pos, move_num):
                    return True
        return False

    def get_score(self):  # Close to P1's goal -> positive, middle -> 0, close to P2's goal -> negative
                          # P1 won -> -1000000000, P2 won -> 1000000000
        if self.winner is None:
            return self.ball[1] - (self.height - 1) // 2
        else:
            return -1000000000 if self.winner is "P1" else 1000000000

    def step(self, move_num, force=True):  # TODO: lite
        self.move_history.append([self.ball[0], self.ball[1], move_num, self.turn])
        if not self._is_edge(self.ball, move_num):
            self._place_edge(self.ball, move_num)
            self.ball = [self.ball[0] + self.move_vectors[move_num][0], self.ball[1] + self.move_vectors[move_num][1]]
            if self.ball in self.P1_goal:
                self.winner = "P2"
            if self.ball in self.P2_goal:
                self.winner = "P1"
        else:
            if force:
                self.winner = "P1" if self.turn is "P2" else "P2"
            else:
                self.winner = "overwriting"
        if not self._is_in_graph(self.ball, last_move_num=move_num):  # swap turn if player didn't bounce
            self.turn = "P1" if self.turn is "P2" else "P2"
        return self.map, self.get_score(), self.winner, self.move_history

    def undo_last_step(self):
        last_turn = self.move_history.pop()
        last_move = self.move_vectors[last_turn[2]]
        self.ball = [self.ball[0] - last_move[0], self.ball[1] - last_move[1]]
        self._erase_edge(self.ball, last_turn[2])
        self.turn = last_turn[3]
        self.winner = None
        if len(self.move_history) >= 1:
            self.last_turn = self.move_history[-1][3]
        else:
            self.last_turn = "P2"

    def play(self, move_nums=None, continue_playing=False, silent=False):
        """Plays the game indefinitely, returns winner"""
        if move_nums:
            for move_num in move_nums:
                map, score, winner, history = self.step(move_num)
                if winner:
                    if not silent:
                        print("Winner is {}".format(winner))
                    return winner
        if move_nums is None or continue_playing:
            while True:
                if not silent:
                    [print(line) for line in self.render_map()]
                if self.turn is "P1":
                    if self.bot_P1 is not None:
                        move = self.bot_P1.get_move()
                    else:
                        move = Game.move_num(input("({},{}) {}:".format(self.ball[0], self.ball[1], self.turn)))
                else:
                    if self.bot_P2 is not None:
                        self.bot_P2.update(self)
                        move = self.bot_P2.get_move()
                    else:
                        move = Game.move_num(input("({},{}) {}:".format(self.ball[0], self.ball[1], self.turn)))
                map, score, winner, history = self.step(move_num=move)
                if winner:
                    if not silent:
                        print("Winner is {}".format(winner))
                    return winner

    def render_map(self):
        map_graph = [list(' '*(3*self.width)) for _ in range(3*self.height)]
        for x in range(len(self.map)):
            for y in range(len(self.map[x])):
                map_graph[y*3+0][x*3+1] = '|' if self._is_edge((x, y), 0) else ' '
                map_graph[y*3+0][x*3+2] = '/' if self._is_edge((x, y), 1) else ' '
                map_graph[y*3+1][x*3+2] = '-' if self._is_edge((x, y), 2) else ' '
                map_graph[y*3+2][x*3+2] = '\\' if self._is_edge((x, y), 3) else ' '
                map_graph[y*3+2][x*3+1] = '|' if self._is_edge((x, y), 4) else ' '
                map_graph[y*3+2][x*3+0] = '/' if self._is_edge((x, y), 5) else ' '
                map_graph[y*3+1][x*3+0] = '-' if self._is_edge((x, y), 6) else ' '
                map_graph[y*3+0][x*3+0] = '\\' if self._is_edge((x, y), 7) else ' '
                map_graph[y*3+1][x*3+1] = '*' if self._is_in_graph([x, y]) else ' '
                if [x, y] in self.P1_goal + self.P2_goal:
                    map_graph[y*3+1][x*3+1] = 'G'
        map_graph[self.ball[1] * 3 + 1][self.ball[0] * 3 + 1] = '1' if self.last_turn is "P1" else '2'
        map_graph = [''.join(line) for line in map_graph]
        return map_graph

    def get_env(self):
        end = np.zeros(shape=(self.height, self.width))
        end[self.ball[1], self.ball[0]] = 1  # one-hot encoded pos
        env = np.transpose(np.array(self.map), axes=(2, 1, 0))  # channels-first, height, width
        env = np.concatenate((env, [end]))
        return env

    def get_winner_history(self):
        assert self.winner in ["P1", "P2"], "game has not ended"
        history = [move[2] for move in self.move_history if move[3] is self.winner]
        one_hot = np.zeros(shape=(len(history), 8), dtype='float16')
        for i in range(len(history)):
            one_hot[i][history[i]] = 1
        return one_hot


if __name__ == "__main__":
    game = Game()
    print(game.get_env())
