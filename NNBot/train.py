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

epochs = 10
games_in_epoch = 100
batch_size = 10
discount_ratio = 0.7
# searchbot_chance = 1
explore = 0.8
test_games = 1000

version = 'v2'

V = Sequential()
V.add(Conv2D(16, (3, 3), input_shape=(9, 8, 7), activation='relu', padding="valid"))
V.add(Conv2D(32, (3, 3), activation='relu', padding="valid"))
V.add(Flatten())
V.add(Dense(256, activation='relu'))
V.add(Dense(128, activation='relu'))
V.add(Dense(64, activation='relu'))
V.add(Dense(8, activation='relu'))
V.add(Dense(1, activation='tanh'))

V.compile(optimizer=Adam(lr=0.1), loss='mse')

for epoch in tqdm(range(epochs)):
    epoch_envs = []
    epoch_rewards = []
    for game_num in range(games_in_epoch):
        game = Game()
        envs = []
        rewards = []
        while True:
            if random.random() < explore:
                _, _, winner, _ = game.step(random.randint(0, 7))
                env = game.get_env()
                envs.append(env)
                rewards.append(V.predict(np.array([env]))[0][0])
            else:
                values = []
                for move_num in range(8):
                    _, _, winner, overwriting = game.step(move_num)
                    if winner:
                        values.append(1 if winner is "P1" else -1)
                    else:
                        values.append(V.predict(np.array([game.get_env()]))[0][0])
                    game.undo_last_step(overwriting)
                # print(np.array(values))
                # print()
                best_value = max(values) if game.turn is "P1" else min(values)
                best_move_num = np.argmax(values) if game.turn is "P1" else np.argmin(values)
                _, _, winner, _ = game.step(best_move_num)
                envs.append(game.get_env())
                rewards.append(best_value)
            if winner:
                break
        for i in reversed(range(len(rewards)-1)):
            rewards[i] += discount_ratio*rewards[i+1]
        assert len(rewards) == len(envs), "len(rewards) and len(envs) not equal"
        epoch_envs.extend(envs)
        epoch_rewards.extend(rewards)
    epoch_envs = np.array(epoch_envs)
    epoch_rewards = np.array(epoch_rewards)
    V.fit(epoch_envs, epoch_rewards, initial_epoch=epoch, callbacks=[TensorBoard(log_dir='./logs')])

    # V.fit
V.save('saves/pencilbot_{}.h5'.format(version))
