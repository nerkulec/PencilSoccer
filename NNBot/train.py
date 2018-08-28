from PencilGame.pencilGame import Game
from PencilGame.bots import SearchBot
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
import random
from tqdm import tqdm
import numpy as np

epochs = 1000
repetitions = 1
limit_moves = 10000
games_in_epoch = 1000
batch_size = 10
searchbot_chance = 1
explore = 0.05
test_games = 1000

version = 'v1.4'

model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=(9, 8, 7), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy')

for epoch in range(epochs):
    winners_envs = []
    winners_moves = []
    if epoch % 5 == 0:
        searchbot_chance /= 1.08
    for game_num in range(games_in_epoch):
        game = Game()
        bot = SearchBot(game=game, max_depth=2)
        P1_envs = []
        P2_envs = []
        try:
            for move_num in range(limit_moves):
                env = game.get_env()
                if game.turn == "P1":
                    P1_envs.append(env)
                if game.turn == "P2":
                    env = np.rot90(env, k=2, axes=(1, 2))
                    P2_envs.append(env)
                else:
                    move = np.argmax(model.predict(np.array([env])))
                if game.turn == "P2":
                    move = (move+4) % 8
                if random.random() < explore:
                    move = random.randrange(8)
                if random.random() < searchbot_chance:
                    bot.update(game)
                    move = bot.get_move()
                game.step(move_num=move)
        except GameOver as e:
            winners_moves.extend(game.get_winner_history())
            if game.history['winner'] is 'P1':
                winners_envs.extend(P1_envs)
            if game.history['winner'] is 'P2':
                winners_envs.extend(P2_envs)
    winners_envs = np.array(winners_envs, dtype='float16')
    winners_moves = np.array(winners_moves, dtype='float16')
    model.fit(winners_envs, winners_moves, epochs=repetitions, batch_size=batch_size)

num_win = 0
for game_num in tqdm(range(test_games)):   # P1 - computer, P2 - random
    game = Game()
    try:
        for move_num in range(limit_moves):
            if game.turn is "P1":
                env = game.get_env()
                move = np.argmax(model.predict(np.array([env])))
            if game.turn is "P2":
                move = random.randrange(8)
            game.step(move_num=move)
    except GameOver as e:
        if game.history['winner'] is "P1":
            num_win += 1

model.save('saves/pencilbot_{}.h5'.format(version))

print("winrate: {}".format(num_win/test_games))
