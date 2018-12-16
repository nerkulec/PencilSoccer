import sys
sys.path.insert(0, '/home/bartek/PythonProjects/PencilSoccer/PencilSoccer')
from PencilGame.pencilGame import Game
from PencilGame.bots import SearchBot
from keras.models import Model
from keras.layers import Conv2D, Flatten, Dense, Input
from keras.optimizers import Adam
from keras import backend as K
import random
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter


version = 'v3.0'

LOSS_CLIPPING = 0.1

GAMMA = 0.99

BUFFER_SIZE = 256
BATCH_SIZE = 64
NUM_ACTIONS = 8
ENTROPY_LOSS = 1e-2
VALUE_LOSS = 1
LR = 1e-4

def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        prob = y_true * y_pred
        old_prob = y_true * old_prediction
        r = prob/(old_prob + 1e-10)
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * (prob * K.log(prob + 1e-10)))
    return loss

class Agent:
    def __init__(self):
        self.model = self.build_model

        self.env = None # TODO: Game()
        print(self.env.action_space, 'action_space', self.env.observation_space, 'observation_space') # TODO
        self.episode = 0
        self.observation = self.env.reset() # TODO
        self.val = False
        self.reward = []
        self.writer = SummaryWriter('pencilGame')
        self.gradient_steps = 0

    def build_model(self):
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(8,))
        state_input = Input(shape=(3*9, 3*8))
        x = Conv2D(32, (3, 3), input_shape=(3*9, 3*8), strides=(3, 3), activation='relu', padding='same')(state_input)
        x = Conv2D(64, (3, 3), activation='relu', padding="valid")(x)
        x = Conv2D(32, (3, 3), activation='relu', padding="valid")(x)
        x = Flatten()(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(32, activation='relu')(x)
        policy = Dense(8, activation='softmax')(x)
        V = Dense(1)(x)

        model = Model(inputs=[state_input, advantage, old_prediction], outputs=[policy, V])
        model.compile(optimizer=Adam(lr=LR),
                      loss=[proximal_policy_optimization_loss(advantage=advantage, old_prediction=old_prediction), 'mse'],
                      loss_weights=[1, VALUE_LOSS])
        model.summary()

        return model

    # Scheme without batching
    model = build_model()
    p, V = model.predict([obs, None, None]) # We don't use advantage and old_prediction in forward pass
    action = random.sample(NUM_ACTIONS, p=p)
    obs, reward, done, info = env.step(action)
    old_prediction = p
    advantage = reward - V
    model.fit([obs, advantage, old_prediction], [one_hot(action, NUM_ACTIONS), reward])

    def reset_env(self):
        self.episode += 1
        if self.episode % 100 == 0:
            self.val = True
        else:
            self.val = False
        self.observation = self.env.reset()
        self.reward = []

    def transform_reward(self):
        if self.val is True:
            self.writer.add_scalar('Val episode reward', np.array(self.reward).sum(), self.episode)
        else:
            self.writer.add_scalar('Episode reward', np.array(self.reward).sum(), self.episode)
        for j in range(len(self.reward) - 2, -1, -1):
            self.reward[j] += self.reward[j + 1] * GAMMA

    def get_batch(self):
        batch = [[], [], [], []]

        tmp_batch = [[], [], []]
        while len(batch[0]) < BUFFER_SIZE:
            if CONTINUOUS is False:
                action, action_matrix, predicted_action = self.get_action()
            else:
                action, action_matrix, predicted_action = self.get_action_continuous()
            observation, reward, done, info = self.env.step(action)
            self.reward.append(reward)

            tmp_batch[0].append(self.observation)
            tmp_batch[1].append(action_matrix)
            tmp_batch[2].append(predicted_action)
            self.observation = observation

            if done:
                self.transform_reward()
                if self.val is False:
                    for i in range(len(tmp_batch[0])):
                        obs, action, pred = tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i]
                        r = self.reward[i]
                        batch[0].append(obs)
                        batch[1].append(action)
                        batch[2].append(pred)
                        batch[3].append(r)
                tmp_batch = [[], [], []]
                self.reset_env()

        obs, action, pred, reward = np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.reshape(np.array(batch[3]), (len(batch[3]), 1))
        pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
        return obs, action, pred, reward

    def run(self):
        while self.episode < EPISODES:
            obs, action, pred, reward = self.get_batch()
            obs, action, pred, reward = obs[:BUFFER_SIZE], action[:BUFFER_SIZE], pred[:BUFFER_SIZE], reward[:BUFFER_SIZE]
            old_prediction = pred
            pred_values = self.critic.predict(obs)

            advantage = reward - pred_values
            # advantage = (advantage - advantage.mean()) / advantage.std()
            actor_loss = self.actor.fit([obs, advantage, old_prediction], [action], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=False)
            critic_loss = self.critic.fit([obs], [reward], batch_size=BATCH_SIZE, shuffle=True, epochs=EPOCHS, verbose=False)
            self.writer.add_scalar('Actor loss', actor_loss.history['loss'][-1], self.gradient_steps)
            self.writer.add_scalar('Critic loss', critic_loss.history['loss'][-1], self.gradient_steps)

            self.gradient_steps += 1


if __name__ == '__main__':
    ag = Agent()
    ag.run()
