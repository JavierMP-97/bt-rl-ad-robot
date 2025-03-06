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

class DQNAgent:
    def __init__(self, state_size, action_size, name="DQNAgent", path="./save/", memory_buffer=16384,   \
    gamma=0.95, epsilon= 1, epsilon_min=0.01, epsilon_decay=0.9995,     \
    learning_rate=0.001, learning_rate_decay=0.0, tau_max=4,        \
    batch_size=32, upgrade_mode=0, model_type=0):
        self.name=name
        self.path=path
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_buffer)
        self.gamma = gamma    # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.tau_max=tau_max
        self.tau=0
        self.upgrade_mode=upgrade_mode
        self.model_type=model_type
        self.batch_size = batch_size
        self.model = self._build_model()
        self.t_model = self._build_model()
        self.update_t_model()
        self.episode=0
        self.tick=0
        self.tick_episode=0

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        if self.model_type == 0:

            model = Sequential()
            model.add(Dense(15, input_dim=self.state_size, activation='relu'))
            model.add(Dense(15, activation='relu'))
            model.add(Dense(15, activation='relu'))
            model.add(Dense(self.action_size, activation='linear')) 
            print(model.summary())

        elif self.model_type == 1:

            inputs = Input(shape=(self.state_size,))
            advt = Dense(20, activation='relu')(inputs)
            advt = Dense(40, activation='relu')(advt)
            advt = Dense(self.action_size)(advt)
            value = Dense(20, activation='relu')(inputs)
            value = Dense(40, activation='relu')(value)
            value = Dense(1)(value)
            # now to combine the two streams
            advt = Lambda(lambda advt: advt - tf.reduce_mean(advt, axis=-1, keepdims=True))(advt)
            value = Lambda(lambda value: tf.tile(value, [1, self.action_size]))(value)
            final = Add()([value, advt])
            model = Model(inputs=inputs, outputs=final)

        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((copy.deepcopy(state), action, reward, copy.deepcopy(next_state), done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size, save_data=True):
        x_batch, y_batch = [], []

        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))

        for state, action, reward, next_state, done in minibatch:

            y_target = self.model.predict(state)
            y_target[0][action] = reward 

            if not done: 
                if self.upgrade_mode == 0: #Normal
                    y_target[0][action] =  reward + self.gamma * np.max(self.model.predict(next_state)[0])
                elif self.upgrade_mode == 1: #Fixed q-target
                    y_target[0][action] =  reward + self.gamma * np.max(self.t_model.predict(next_state)[0])
                elif self.upgrade_mode == 2: #Double q-network
                    q_next_state = np.argmax(self.model.predict(next_state)[0])
                    y_target[0][action] =  reward + self.gamma * self.t_model.predict(next_state)[0, q_next_state]

            x_batch.append(state[0])
            y_batch.append(y_target[0])
        
        loss = self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), epochs=1, verbose=0)
        print(loss)
        #if save_data:
            
        
        self.tau += 1
        if self.tau>self.tau_max:
            self.tau=0
            self.update_t_model()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, tupla):
        state, action, reward, next_state, done = tupla
        q_next_state = np.argmax(self.model.predict(next_state)[0])
        target = reward
        if not done:
            if self.upgrade_mode == 0: #Normal
                target =  reward + self.gamma * np.max(self.model.predict(next_state)[0])
            elif self.upgrade_mode == 1: #Fixed q-target
                target =  reward + self.gamma * np.max(self.t_model.predict(next_state)[0])
            elif self.upgrade_mode == 2: #Double q-network
                q_next_state = np.argmax(self.model.predict(next_state)[0])
                target =  reward + self.gamma * self.t_model.predict(next_state)[0, q_next_state]
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, batch_size=1, epochs=1, verbose=0) 

        self.tau += 1
        if self.tau>self.tau_max:
            self.tau=0
            self.update_t_model()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def pre_train(self, lista, epochs):
        for e in range(epochs):
            x_batch, y_batch = [], []

            minibatch = random.sample(lista, min(len(lista),  int((e+1)*len(lista)/10)))

            for state, action, reward, next_state, done in minibatch:
                if action == 7:
                    action = 3

                y_target = self.model.predict(state)
                y_target[0][action] = reward 

                if not done: 
                    if self.upgrade_mode == 0: #Normal
                        y_target[0][action] =  reward + self.gamma * np.max(self.model.predict(next_state)[0])
                    elif self.upgrade_mode == 1: #Fixed q-target
                        y_target[0][action] =  reward + self.gamma * np.max(self.t_model.predict(next_state)[0])
                    elif self.upgrade_mode == 2: #Double q-network
                        q_next_state = np.argmax(self.model.predict(next_state)[0])
                        y_target[0][action] =  reward + self.gamma * self.t_model.predict(next_state)[0, q_next_state]

                x_batch.append(state[0])
                y_batch.append(y_target[0])
            
            loss = self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), epochs=1, verbose=0)
            print(loss)


    def update_t_model(self):
        self.t_model.set_weights(self.model.get_weights())


    def load(self):
        try:
            with open(self.path+self.name+".pr", "rb") as f:
                d = pickle.load(f)
            self.epsilon = d["epsilon"]
            self.epsilon = 0.3
            self.memory.extend(d["memory"])
            self.tau = d["tau"]
            self.model.load_weights(self.path+self.name+".hdf5")
            self.update_t_model()
        except:
            print("{} couldn't load. Creating a new {}\n".format(self.path+self.name, self.name))
            self.save()
        else:
            print("{} loaded\n".format(self.path+self.name))
        #self.t_model.load_weights(self.path+self.name+"_t.hdf5")
        

    def save(self):
        d = {}
        d["epsilon"] = self.epsilon
        d["memory"] = self.memory
        d["tau"] = self.tau
        with open( self.path+self.name+".pr" , "wb" ) as f:
            pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL )
        self.model.save_weights(self.path+self.name+".hdf5")

        print("{} saved\n".format(self.path+self.name))
        #self.t_model.save_weights(self.path+self.name+"_t.hdf5")