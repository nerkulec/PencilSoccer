import sys
sys.path.insert(0, '/home/bartek/PythonProjects/PencilSoccer/PencilSoccer')
from PencilGame.pencilGame import Game
from PencilGame.bots import SearchBot
from keras.models import Model
from keras.layers import Conv2D, Flatten, Dense, Input
from keras.optimizers import Adam
from keras import backend as K
import gym
import random
from tqdm import tqdm
import numpy as np


version = 'v3.0'

LOSS_CLIPPING = 0.1

GAMMA = 0.99

EPISODES = 10
EPOCHS = 10

BUFFER_SIZE = 256
BATCH_SIZE = 64
NUM_ACTIONS = 8
ENTROPY_LOSS = 1e-2
VALUE_LOSS = 1
LR = 1e-4

def proximal_policy_optimization_loss(advantage, old_distribution):
    def loss(y_true, y_pred):
        prob = y_true * y_pred
        old_prob = y_true * old_distribution
        r = prob/(old_prob + 1e-10)
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * (prob * K.log(prob + 1e-10)))
    return loss

class Agent:
    def __init__(self):
        self.model = self.build_model()
        self.env = gym.make('CartPole-v0')
        print(self.env.action_space, 'action_space', self.env.observation_space, 'observation_space') # TODO
        self.episode = 0
        self.observation = self.env.reset() # TODO
        self.reward = []
        self.gradient_steps = 0

    def build_model(self):
        # advantage = Input(shape=(1,), name='advantage')
        # old_distribution = Input(shape=(8,), name='old_distribution')
        # state_input = Input(shape=(3*9, 3*8), name='state_input')
        # x = Conv2D(32, (3, 3), strides=(3, 3), activation='relu', padding='same')(state_input)
        # x = Conv2D(64, (3, 3), activation='relu', padding="valid")(x)
        # x = Conv2D(32, (3, 3), activation='relu', padding="valid")(x)
        # x = Flatten()(x)
        # x = Dense(64, activation='relu')(x)
        # x = Dense(32, activation='relu')(x)
        # policy = Dense(8, activation='softmax')(x)
        # V = Dense(1)(x)

        state_input = Input(shape=(4,), name='state_input')
        advantage = Input(shape=(1,), name='advantage')
        old_distribution = Input(shape=(2,), name='old_distribution')
        x = Dense(32, activation='relu')(state_input)
        x = Dense(64, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        policy = Dense(2, activation='softmax')(x)
        V = Dense(1)(x)

        model = Model(inputs=[state_input, advantage, old_distribution], outputs=[policy, V])
        model.compile(optimizer=Adam(lr=LR),
                      loss=[proximal_policy_optimization_loss(advantage=advantage, old_distribution=old_distribution), 'mse'],
                      loss_weights=[1, VALUE_LOSS])
        model.summary()

        return model

    def reset_env(self):
        self.episode += 1
        self.observation = self.env.reset()
        self.reward = []

    def get_action(self):
        p, V = self.model.predict([self.observation.reshape(1, 4), np.array([2]), np.array([[1, 2]])])
        action = np.random.choice(2, p=np.nan_to_num(p[0]))
        action_matrix = np.zeros(p[0].shape)
        action_matrix[action] = 1
        return action, action_matrix, p, V

    def transform_reward(self):
        for j in range(len(self.reward) - 2, -1, -1):
            self.reward[j] += self.reward[j + 1] * GAMMA

    def get_batch(self):
        batch = [[], [], [], [], []]

        tmp_batch = [[], [], [], []]
        while len(batch[0]) < BUFFER_SIZE:
            action, action_matrix, predicted_action, predicted_value = self.get_action()
            observation, reward, done, info = self.env.step(action)
            self.reward.append(reward)

            tmp_batch[0].append(self.observation)
            tmp_batch[1].append(action_matrix)
            tmp_batch[2].append(predicted_action)
            tmp_batch[3].append(predicted_value)
            self.observation = observation

            if done:
                self.transform_reward()
                for i in range(len(tmp_batch[0])):
                    obs, action, pred, pred_val = tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i], tmp_batch[3][i]
                    r = self.reward[i]
                    batch[0].append(obs)
                    batch[1].append(action)
                    batch[2].append(pred)
                    batch[3].append(r)
                    batch[4].append(pred_val)
                tmp_batch = [[], [], [], []]
                self.reset_env()

        obs, action, pred, reward, pred_val = np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.array(batch[3]), np.array(batch[4])
        pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
        reward = np.reshape(reward, (len(batch[3]), 1))
        pred_val = np.reshape(pred_val, (len(batch[4]), 1))
        return obs, action, pred, reward, pred_val

    def run(self):
        while self.episode < EPISODES:
            obs, action, pred, reward, pred_values = self.get_batch()
            obs, action, pred, reward, pred_values = obs[:BUFFER_SIZE], action[:BUFFER_SIZE], pred[:BUFFER_SIZE], reward[:BUFFER_SIZE], pred_values[:BUFFER_SIZE]
            old_distribution = pred

            advantage = reward - pred_values
            # advantage = (advantage - advantage.mean()) / advantage.std()
            history = self.model.fit([obs, advantage, old_distribution], [action, reward], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=False)
            loss = history.history['loss'][-1]
            print(loss)

if __name__ == '__main__':
    ag = Agent()
    ag.run()
