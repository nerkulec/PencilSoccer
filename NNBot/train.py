from PencilGame.pencilGame import Game
from PencilGame.bots import SearchBot
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Input
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.utils import plot_model
import random
from tqdm import tqdm
import numpy as np

epochs = 1000
games_in_epoch = 10
batch_size = 10
discount_ratio = 0.9
searchbot_chance = 0.7
explore = 0
test_games = 1000

version = 'v2.1'

V = Sequential()
V.add(Conv2D(32, (3, 3), input_shape=(9, 8, 7), activation='relu', padding="valid"))
V.add(Conv2D(64, (3, 3), activation='relu', padding="valid"))
V.add(Flatten())
V.add(Dense(256, activation='relu'))
V.add(Dense(128, activation='relu'))
V.add(Dense(64, activation='relu'))
V.add(Dense(8, activation='relu'))
V.add(Dense(1, activation='tanh'))

V.compile(optimizer=Adam(lr=0.001), loss='mse')

for epoch in tqdm(range(epochs)):
    epoch_envs = []
    epoch_rewards = []
    for game_num in range(games_in_epoch):
        game = Game()
        searchbot = SearchBot(game, 3)
        envs = []
        rewards = []
        while True:
            if random.random() < explore:
                _, _, winner, _ = game.step(random.randint(0, 7))
                env = game.get_env()
                envs.append(env)
                if winner:
                    rewards.append(1 if winner is "P1" else -1)
                else:
                    rewards.append(0)
            elif random.random() < searchbot_chance:
                _, _, winner, _ = game.step(searchbot.get_move())
                env = game.get_env()
                envs.append(env)
                if winner:
                    rewards.append(1 if winner is "P1" else -1)
                else:
                    rewards.append(0)
            else:
                values = []
                for move_num in range(8):
                    _, _, winner, overwriting = game.step(move_num)
                    values.append(V.predict(np.array([game.get_env()]))[0][0])
                    game.undo_last_step(overwriting)
                best_value = max(values) if game.turn is "P1" else min(values)
                best_move_num = np.argmax(values) if game.turn is "P1" else np.argmin(values)
                _, _, winner, _ = game.step(best_move_num)
                envs.append(game.get_env())
                if winner:
                    rewards.append(1 if winner is "P1" else -1)
                else:
                    rewards.append(0)
            if winner:
                break
        for i in reversed(range(len(rewards)-1)):
            rewards[i] += discount_ratio*rewards[i+1]
        assert len(rewards) == len(envs), "len(rewards) and len(envs) not equal"
        epoch_envs.extend(envs)
        epoch_rewards.extend(rewards)
    epoch_envs = np.array(epoch_envs)
    epoch_rewards = np.array(epoch_rewards)
    V.train_on_batch(epoch_envs, epoch_rewards)

V.save('saves/pencilbot_{}.h5'.format(version))
