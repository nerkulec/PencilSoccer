from PencilGame.pencilGame import Game
from PencilGame.bots import RandomBot, NNBot, SearchBot
# from keras.models import load_model

game = Game()
game.set_bot_P2(SearchBot(game, max_depth=9))
game.set_bot_P1(SearchBot(game, max_depth=9))
game.play()
