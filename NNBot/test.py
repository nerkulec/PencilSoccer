from PencilGame.pencilGame import GameOver, Game
from PencilGame.bots import RandomBot, NNBot, SearchBot
# from keras.models import load_model
import numpy as np

game = Game()
game.set_bot_P2(SearchBot(game, max_depth=6))
game.set_bot_P1(SearchBot(game, max_depth=6))
game.play()
