# -*- coding: utf-8 -*-
import random
import gym
import gym_colonizer
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM,CuDNNLSTM
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.utils import plot_model

EPISODES = 


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        #self.model = self.build_lstm_model()#self._build_model()
        #self.target_model = self.build_lstm_model()#_build_model()
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    """Huber loss for Q Learning

    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        embed_dim = 130 # old 64
        model.add(Dense(embed_dim, input_dim=self.state_size, activation='relu'))
        model.add(Dense(embed_dim, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        plot_model(model, to_file='model.png')
        return model
    
    def build_lstm_model(self):
        embed_dim = 128
        lstm_out = 200
        batch_size = 32

        model = Sequential()
        model.add(Embedding(100, embed_dim,input_length = self.state_size, dropout = 0.2))
        model.add(CuDNNLSTM(lstm_out))
        model.add(Dense(self.action_size,activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
        print(model.summary())
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

inputs = 129
n_actions = 54+72+1 # build spots+roads+ratings+doNothing

if __name__ == "__main__":
    s = tf.compat.v1.Session()
    print(s.list_devices())
    env = gym.make('Colonizer-v0')
    state_size = inputs
    action_size = n_actions
    agent = DQNAgent(state_size, action_size)
    #agent.load("./save/agent-ddqn.h5")
    done = False
    batch_size = 32

    rewards = {}
    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        rewards[e] = 0
        for time in range(40):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action,time)
            reward = reward if not done else -10
            rewards[e] += reward
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done or time == 19:
                agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, rewards[e], agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % 1000 == 0:
            agent.save("./save/agent-ddqn.h5")
