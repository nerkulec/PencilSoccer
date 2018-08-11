import numpy as np


class GameOver(Exception):
    def __init__(self, message, winner=None):
        super().__init__(message)
        self.winner = winner
        self.message = message


class Game:
    def __init__(self, width=7, height=8, lite=False):
        self.width = width
        self.height = height
        self.bot_P1 = None
        self.bot_P2 = None
        self.lite = lite
        self.moves = [[0, -1], [1, -1], [1, 0], [1, 1],
                      [0, 1], [-1, 1], [-1, 0], [-1, -1]]  # UP-> 0, then clockwise
        self.possible_move_nums = [0, 1, 2, 3, 4, 5, 6, 7]
        self.map = self.gen_map()
        self.ball = [self.width // 2, (self.height - 1) // 2]
        self.turn = "P1"
        self.last_turn = "P1"
        self.P1_goal = [[self.width//2-1, 0],
                        [self.width//2, 0],
                        [self.width//2+1, 0]]
        self.P2_goal = [[self.width//2-1, self.height-1],
                        [self.width//2, self.height-1],
                        [self.width//2+1, self.height-1]]
        self.history = {
            'width': width,
            'height': height,
            'lite': lite,
            'start_pos': self.ball,
            'starting_player': self.turn,
            'P1_goal': self.P1_goal,
            'P2_goal': self.P2_goal,
            'game_ended': False,
            'winner': None,
            'win_cause': None,
            'move_nums': [],
            'turns': []
        }

    def set_bot_P1(self, bot):
        self.bot_P1 = bot

    def set_bot_P2(self, bot):
        self.bot_P2 = bot

    def copy(self):
        game_copy = Game(width=self.width, height=self.height, lite=self.lite)
        for move in self.history['move_nums']:  # TODO: lite
            game_copy.move(move_num=move)
        return game_copy

    def gen_map(self):
        full_map = np.zeros(shape=(self.width, self.height, len(self.moves)))
        full_map[:, 0, :3] = 1
        full_map[:, 0, -2:] = 1
        full_map[:, self.height-1, 2:7] = 1
        full_map[self.width-1, :, 0:5] = 1
        full_map[0, :, 4:] = 1
        full_map[0, :, 0] = 1
        return full_map

    @staticmethod
    def move_num(move):
        return {"u": 0, "ur": 1, "ru": 1, "r": 2, "dr": 3, "rd": 3,
                "d": 4, "dl": 5, "ld": 5, "l": 6, "ul": 7, "lu": 7}[move]

    def _place_edge(self, pos, move_num):  # TODO: lite
        self.map[pos[0], pos[1], move_num] = 1
        self.map[pos[0] + self.moves[move_num][0], pos[1] + self.moves[move_num][1], (move_num + 4) % 8] = 1  # opposite side

    def _is_edge(self, pos, move_num):  # TODO: lite
        return self.map[pos[0], pos[1], move_num] == 1

    def _is_escapable(self, pos, except_move_num=None):
        for move_num in range(len(self.moves)):
            if not self._is_edge(pos, move_num) and (not except_move_num or move_num != (except_move_num+4) % 8):
                return True
        return False

    def _is_in_graph(self, pos, last_move_num=None):
        """ Returns True if edge is in graph. If last_move_num is specified ignores last move """
        for move_num in self.possible_move_nums:
            if last_move_num is None or move_num != (last_move_num + 4) % 8:
                if self._is_edge(pos, move_num):
                    return True
        return False

    def move(self, move_num):  # TODO: lite
        self.last_turn = self.turn
        self.history['move_nums'].append(move_num)
        self.history['turns'].append(self.turn)
        if not self._is_edge(self.ball, move_num):
            self._place_edge(self.ball, move_num)
            self.ball = [self.ball[0] + self.moves[move_num][0], self.ball[1] + self.moves[move_num][1]]  # move
            if self.ball in self.P1_goal:
                self.history['winner'] = "P2"
                self.history['win_cause'] = "goal"
                self.history['game_ended'] = True
                raise GameOver("Ball in P1's goal", winner="P2")
            if self.ball in self.P2_goal:
                self.history['winner'] = "P1"
                self.history['win_cause'] = "goal"
                self.history['game_ended'] = True
                raise GameOver("Ball in P2's goal", winner="P1")
            if not self._is_escapable(self.ball):
                self.history['winner'] = "P1" if self.turn is "P2" else "P2"
                self.history['win_cause'] = "inescapable"
                self.history['game_ended'] = True
                raise GameOver("Inescapable", winner="P1" if self.turn is "P2" else "P2")
        else:
            self.history['winner'] = "P1" if self.turn is "P2" else "P2"
            self.history['win_cause'] = "overwriting"
            self.history['game_ended'] = True
            raise GameOver("Writing over line", winner="P1" if self.turn is "P2" else "P2")
        if not self._is_in_graph(self.ball, move_num):  # swap turn if player didn't bounce
            self.turn = "P1" if self.turn is "P2" else "P2"

    def test(self, move_num):
        if not self._is_edge(self.ball, move_num):
            test_ball = [self.ball[0] + self.moves[move_num][0], self.ball[1] + self.moves[move_num][1]]
            if test_ball in self.P1_goal:
                raise GameOver("Ball in P1's goal", winner="P2")
            if test_ball in self.P2_goal:
                raise GameOver("Ball in P2's goal", winner="P1")
            if not self._is_escapable(test_ball, except_move_num=move_num):
                raise GameOver("Inescapable", winner="P1" if self.turn is "P2" else "P2")
        else:
            raise GameOver("Writing over line", winner="P1" if self.turn is "P2" else "P2")

    def play(self, move_nums=None, continue_playing=False, silent=False):
        """Plays the game indefinitely, returns winner"""
        if move_nums:
            for move_num in move_nums:
                try:
                    self.move(move_num)
                except GameOver as e:
                    if not silent:
                        print("Winner is {}".format(e.winner))
                        print(e.message)
                    return e.winner, self.history['win_cause']
        if move_nums is None or continue_playing:
            while True:
                if not silent:
                    [print(line) for line in self.render_map()]
                if self.turn is "P1":
                    if self.bot_P1 is not None:
                        self.bot_P1.update(self)
                        move = self.bot_P1.get_move()
                    else:
                        move = Game.move_num(input("({},{}) {}:".format(self.ball[0], self.ball[1], self.turn)))
                else:
                    if self.bot_P2 is not None:
                        self.bot_P2.update(self)
                        move = self.bot_P2.get_move()
                    else:
                        move = Game.move_num(input("({},{}) {}:".format(self.ball[0], self.ball[1], self.turn)))
                try:
                    self.move(move_num=move)
                except GameOver as e:
                    if not silent:
                        print("Winner is {}".format(e.winner))
                        print(e.message)
                    return e.winner, self.history['win_cause']

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
        assert self.history['game_ended'] is True, "game has not ended"
        history = [move for (i, move) in enumerate(self.history['move_nums'])
                   if self.history['turns'][i] == self.history['winner']]
        one_hot = np.zeros(shape=(len(history), 8), dtype='float16')
        for i in range(len(history)):
            one_hot[i][history[i]] = 1
        return one_hot


if __name__ == "__main__":
    game = Game()
    print(game.get_env())
    while True:
        try:
            [print(line) for line in game.render_map()]
            move = input("({},{}) {}:".format(game.ball[0], game.ball[1], game.turn))
            game.move(move_num=game.move_num(move))
        except KeyError as e:
            print(e)
        except GameOver as e:
            print("Winner is {}".format(e.winner))
            print(e.message)
            break
