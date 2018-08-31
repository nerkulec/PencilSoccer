from PencilGame.pencilGame import Game
from PencilGame.bots import RandomBot, NNBot, SearchBot
from keras.models import load_model

game = Game(value_model=load_model('saves/pencilbot_v2.h5'))
# game.set_bot_P1()
game.set_bot_P2(NNBot(game, 'v2'))
game.play()
