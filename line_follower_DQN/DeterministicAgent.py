import socket
import time
import random
import gym
import numpy as np
import pickle
import copy
from collections import deque
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Lambda
from keras.optimizers import Adam
from keras.layers.merge import Add

import tensorflow as tf
from keras import backend as K


def huber_loss(a, b, in_keras=True):
    error = a - b
    quadratic_term = error*error / 2
    linear_term = abs(error) - 1/2
    use_linear_term = (abs(error) > 1.0)
    if in_keras:
        # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
        use_linear_term = K.cast(use_linear_term, 'float32')
    return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term

class DeterministicAgent:
    def __init__(self, state_size, action_size, name="DeterministicAgent", path="./save/", memory_buffer=2048,   \
    gamma=0.95, epsilon= 1, epsilon_min=0.01, epsilon_decay=0.9995,     \
    learning_rate=0.001, learning_rate_decay=0.0, tau_max=4,        \
    batch_size=32, upgrade_mode=0, model_type=0):
        self.name=name
        self.path=path
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.episode=0
        self.tick=0
        self.tick_episode=0
        self.last_line_position=0.5

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((copy.deepcopy(state), action, reward, copy.deepcopy(next_state), done))

    def act(self, data, limit):
        line_position = np.argmin(data)

        if data[line_position] > (limit[line_position]*1.5):
            if self.last_line_position == 0:
                action = 0
            elif self.last_line_position == 1:
                action = 7
            elif self.last_line_position == 2:
                action = 6

        else:
            if line_position < 1:
                self.last_line_position = 0
            elif line_position > 3:
                self.last_line_position = 2
            else:
                self.last_line_position = 1

            action = line_position + 1
        return action
        

    def save(self):
        d = {}
        d["memory"] = self.memory
        pickle.dump( d, open( self.path+self.name+".pr" , "wb" ), protocol=pickle.HIGHEST_PROTOCOL )

        print("{} saved\n".format(self.path+self.name))
        #self.t_model.save_weights(self.path+self.name+"_t.hdf5")