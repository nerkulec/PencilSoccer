from PencilGame.pencilGame import Game, GameOver
from PencilGame.bots import SearchBot
import time

while True:
    game = Game()
    P1_bot = SearchBot(game=game, max_depth=4)
    P2_bot = SearchBot(game=game, max_depth=5)
    while True:
        [print(line) for line in game.render_map()]
        try:
            if game.turn is "P1":
                P1_bot.update(game)
                move = P1_bot.get_move()
            else:
                P2_bot.update(game)
                move = P2_bot.get_move()
            game.step(move_num=move)
        except GameOver as e:
            print("Winner is {}".format(e.winner))
            print(e.message)
            break
