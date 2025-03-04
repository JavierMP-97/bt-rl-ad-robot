
import socket
import time
import random
import gym
import numpy as np
import pickle
import copy
import math
from collections import deque

import matplotlib.pyplot as plt
import pandas as pd

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Lambda, Convolution2D, MaxPooling2D, Cropping3D, Conv3D, MaxPooling3D, BatchNormalization, LeakyReLU, Cropping2D, Conv2D
from keras.activations import relu
from keras.optimizers import Adam
from keras.layers.merge import Add, Concatenate

import tensorflow as tf
from keras import backend as K

    #Usar/no usar GPU

num_cores = 4

GPU=True

if GPU:
    num_GPU = 1
    num_CPU = 1
    print("Using GPU\n")
else:
    num_CPU = 1
    num_GPU = 0
    print("Not using GPU\n")

config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,\
        inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
        device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})

session = tf.Session(config=config)
K.set_session(session)

print("\nSession loaded\n")

def huber_loss(a, b, in_keras=True):
    error = a - b
    quadratic_term = error*error / 2
    linear_term = abs(error) - 1/2
    use_linear_term = (abs(error) > 1.0)
    if in_keras:
        # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
        use_linear_term = K.cast(use_linear_term, 'float32')
    return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term

def normalized_variance(prediction):
    max_v = np.max(prediction)
    min_v = np.min(prediction)

    norm_pred = [ (p - min_v) / (max_v - min_v) for p in prediction[0] ]

    variance = np.var(norm_pred)

    return variance

def random_action(action):
    distance_1 = [[1,0],[0,1]]
    distance_2 = [[2,0],[0,2],[1,1]]
    distance_3 = [[3,0],[2,1],[1,2]]
    distance_4 = [[4,0],[3,1],[4,1],[2,2],[3,2],[4,2]]

    y = math.floor(action / 5)
    x = action - y*5

    action_distance = np.random.rand()
    if action_distance <= 0.4:
        actions_distance = distance_1
    elif action_distance <= 0.7:
        actions_distance = distance_2
    elif action_distance <= 0.9:
        actions_distance = distance_3
    else:
        return random.randrange(15)
        #actions_distance = distance_4

    action_modifier = random.choice(actions_distance)
    x_sign = random.randrange(-1,2,2)
    y_sign = random.randrange(-1,2,2)
    new_x = x + (action_modifier[0] * x_sign)
    new_y = y + (action_modifier[1] * y_sign)
    while new_x < 0 or new_x >= 5 or new_y < 0 or new_y >= 3:
        action_modifier = random.choice(actions_distance)
        x_sign = random.randrange(-1,2,2)
        y_sign = random.randrange(-1,2,2)
        new_x = x + (action_modifier[0] * x_sign)
        new_y = y + (action_modifier[1] * y_sign)

    return new_x + 5*new_y

class DQNAgent:
    def __init__(self, state_size, action_size, name="Replay64x10-end-ep_DQNAgent", path="./save/", memory_buffer=10000,   \
    gamma=0.99, epsilon=1, epsilon_min=0.1, epsilon_decay=0.9997,     \
    learning_rate=0.00001, learning_rate_decay=0.0, tau_max=100,        \
    batch_size=64, upgrade_mode=2, model_type=0):
        
        self.name = name
        self.path = path

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
        self.tau = 0
        self.upgrade_mode=upgrade_mode
        self.model_type=model_type
        self.batch_size = batch_size
        
        self.model = self._build_model()
        self.model.summary()
        self.t_model = self._build_model()
        self.update_t_model()
        self.best_model = self._build_model()
        self.best_model_number = 0
        
        self.episode = 0
        self.tick = 0
        self.tick_episode = 0
        self.acum_loss = 0
        self.acum_reward = 0
        self.acum_variance = 0
        self.best_reward = 0

        self.update_best_model()

        with open(self.path+self.name+".txt", "a") as info:
            info.write("episode,ticks,epsilon,loss,acum_reward,avg_reward,total_ticks\n")
            info.close()
        with open(self.path+self.name+"_test.txt", "a") as info:
            info.write("episode,ticks,acum_reward,avg_reward\n")
            info.close()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        if self.model_type == 0:

            #inputs = Input(shape=(self.state_size[0], self.state_size[1], self.state_size[2], 1))
            #crop = Cropping3D(cropping=((0,0),(40,0),(0,0)))(inputs)
            #conv = Conv3D(filters=8, kernel_size=(3, 3, 3), strides=(1, 2, 2), kernel_initializer="he_normal")(crop)
            inputs1 = Input(shape=(self.state_size[1], self.state_size[2], self.state_size[3],), name="image")
            #crop = Cropping2D(cropping=((40,0),(0,0)))(inputs1)
            conv = Conv2D(filters=8, kernel_size=(3, 3), strides=(2, 2), kernel_initializer="he_normal")(inputs1) #, kernel_initializer="he_normal"
            conv = LeakyReLU(alpha=0.0)(conv)
            #conv = MaxPooling3D(pool_size=(1,2,2))(conv)
            conv = MaxPooling2D(pool_size=(2,2))(conv)
            #conv = BatchNormalization()(conv)
            #conv = Dropout(0.1)(conv)
            conv = Flatten()(conv)
            inputs2 = Input(shape=(1,), name="speed")
            #speed = Dense(10, kernel_initializer="he_normal")(inputs2)
            mlp = Concatenate()([conv, inputs2])
            mlp = Dense(50, kernel_initializer="he_normal")(mlp) #, kernel_initializer="he_normal"
            mlp = LeakyReLU(alpha=0.0)(mlp)
            #mlp = Dense(50, kernel_initializer="he_normal")(mlp)
            #mlp = LeakyReLU(alpha=0.0)(mlp)
            #mlp = Dense(50, kernel_initializer="he_normal")(mlp)
            #mlp = LeakyReLU(alpha=0.0)(mlp)
            #mlp = Dropout(0.5)(mlp)
            final = Dense(self.action_size, activation='linear')(mlp)

            model = Model(inputs=[inputs1, inputs2], outputs=final)

        elif self.model_type == 1:

            #inputs = Input(shape=(self.state_size[0], self.state_size[1], self.state_size[2], 1))
            #crop = Cropping3D(cropping=((0,0),(40,0),(0,0)))(inputs)
            #conv = Conv3D(filters=8, kernel_size=(3, 3, 3), strides=(1, 2, 2), kernel_initializer="he_normal")(crop)
            inputs1 = Input(shape=(self.state_size[1], self.state_size[2], self.state_size[3],), name="image")
            #crop = Cropping2D(cropping=((40,0),(0,0)))(inputs1)
            conv = Conv2D(filters=8, kernel_size=(3, 3), strides=(2, 2), kernel_initializer="he_normal")(inputs1) #, kernel_initializer="he_normal"
            conv = LeakyReLU(alpha=0.0)(conv)
            #conv = MaxPooling3D(pool_size=(1,2,2))(conv)
            conv = MaxPooling2D(pool_size=(2,2))(conv)
            #conv = BatchNormalization()(conv)
            #conv = Dropout(0.1)(conv)
            conv = Flatten()(conv)
            inputs2 = Input(shape=(1,), name="speed")
            mlp = Concatenate()([conv, inputs2])
            mlp = Dense(50, kernel_initializer="he_normal")(mlp) #, kernel_initializer="he_normal"
            mlp = LeakyReLU(alpha=0.0)(mlp)
            #mlp = Dense(50, kernel_initializer="he_normal")(mlp)
            #mlp = LeakyReLU(alpha=0.0)(mlp)
            #mlp = Dense(50, kernel_initializer="he_normal")(mlp)
            #mlp = LeakyReLU(alpha=0.0)(mlp)
            #model.add(Dropout(0.2))
            final = Dense(self.action_size+1, activation='linear')(mlp)
            final = Lambda(lambda i: K.expand_dims(i[:,0],-1) + i[:,1:] - K.mean(i[:,1:], keepdims=True), output_shape=(self.action_size,))(final)

            model = Model(inputs=[inputs1, inputs2], outputs=final)

        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate, decay = self.learning_rate_decay , clipnorm=1.))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((copy.deepcopy(state), action, reward, copy.deepcopy(next_state), done))

    def act(self, state, train=True):
        act_values = self.model.predict(state)
        action = np.argmax(act_values[0])
        print("Q-values: {}".format(act_values))
        if train and np.random.rand() <= self.epsilon:
            if self.epsilon > 0.0:    
                return random.randrange(self.action_size)
            else:
                return random_action(action)

        #print("Normalized variance: {}".format(normalized_variance(act_values)))
        return action # returns action

    def replay(self, batch_size, save_data=True):
        x_batch1, x_batch2, y_batch = [], [], []

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

            x_batch1.append(state[0][0])
            x_batch2.append(state[1][0])
            
            y_batch.append(y_target[0])
        
        xbatch1 = np.asarray(x_batch1)
        xbatch2 = np.asarray(x_batch2)
        ybatch = np.asarray(y_batch)
        print(xbatch1.shape)
        loss = self.model.fit([xbatch1, xbatch2], ybatch, batch_size=64, epochs=1, verbose=0)
        
        #with open(self.path+self.name+"_loss.txt", "a") as info:
        #    print("Loss: {}\n".format(loss.history["loss"][0]))
        #    info.write("{}\n".format(loss.history["loss"][0]))
        #    info.close()
        #self.acum_loss += loss.history["loss"][0]
        
        self.update_target()

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
        loss = self.model.fit(state, target_f, batch_size=1, epochs=1, verbose=0) 
        
        
        
        with open(self.path+self.name+"_loss.txt", "a") as info:
            print("Loss: {}\n".format(loss.history["loss"][0]))
            info.write("{}\n".format(loss.history["loss"][0]))
            info.close()
        self.acum_loss += loss.history["loss"][0]
        
        print("Target Q: {}\nPredicted Q:{}\n".format(target_f, self.model.predict(state)))
        
        self.update_target()
        

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
            
            self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), epochs=1, verbose=0)
            


    def update_t_model(self):
        self.t_model.set_weights(self.model.get_weights())
        print("Target model updated\n")
    
    def update_best_model(self):
        '''
        if self.acum_reward > self.best_reward and self.acum_reward/self.tick_episode>0.1:
            self.best_reward = self.acum_reward
            self.best_model.set_weights(self.model.get_weights())
            print("Best model updated\n")
            
        if self.acum_reward > 250:
            self.best_model.save_weights(self.path+self.name+str(self.best_model_number)+"_best.hdf5")
            self.best_model_number += 1
        '''
        if  self.acum_reward > 250 and self.acum_reward/self.tick_episode>0.2:
            self.model.save_weights(self.path+self.name+"_"+str(self.acum_reward)+"_"+str(self.acum_reward/self.tick_episode)+".hdf5")

    def update_target(self):
        if self.tau_max <= 0:
            print("Tau mus be greater than 0")
        elif self.tau_max >= 1:
            self.tau += 1
            if self.tau>=self.tau_max:
                self.tau=0
                self.update_t_model()
                #self.tau_max = self.tau_max * 0.99

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        else:
            print("Not working")

    def load(self):
        try:          
            with open(self.path+self.name+".pr", "rb") as f:
                d = pickle.load(f)
            self.epsilon = d["epsilon"]
            self.episode = d["episode"]
            self.tick = d["tick"]
            #self.best_model_number = d["best_model_number"]
            #self.memory.extend(d["memory"])
            #self.tau = d["tau"]
            
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
        d["episode"] = self.episode
        d["tick"] = self.tick
        #d["best_model_number"] = self.best_model_number
        #d["memory"] = self.memory
        #d["tau"] = self.tau
        with open( self.path+self.name+".pr" , "wb" ) as f:
            pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL )
        
        self.model.save_weights(self.path+self.name+".hdf5")

        print("{} saved\n".format(self.path+self.name))
        #self.t_model.save_weights(self.path+self.name+"_t.hdf5")

    def next_episode(self, train=True):
        if train:
            print("{} finished\n".format(self.path+self.name))
            print("Episode: {}. Ticks: {}. Epsilon: {}. Total ticks: {}. Loss:{}. Acum_Reward: {}. Avg_Reward: {}\n".format(self.episode, self.tick_episode, self.epsilon, self.tick, self.acum_loss/self.tick_episode, self.acum_reward, self.acum_reward/self.tick_episode))

            with open(self.path+self.name+".txt", "a") as info:
                info.write("{},{},{},{},{},{},{}\n".format(self.episode, self.tick_episode, self.epsilon, self.acum_loss/self.tick_episode, self.acum_reward, self.acum_reward/self.tick_episode, self.tick))
                info.close()
                plt.plot()
            
            grafica = pd.read_csv(self.path+self.name+".txt")

            plt.plot(grafica["ticks"])
            plt.savefig(self.path+self.name+"_ticks.png")
            plt.clf()
            plt.plot(grafica["acum_reward"])
            plt.savefig(self.path+self.name+"_reward.png")
            plt.clf()
            plt.plot(grafica["loss"])
            plt.savefig(self.path+self.name+"_loss.png")
            plt.clf()

            
            if self.tick>=2048:
                print("Replay started...\n")
                for q in range(1):
                    self.replay(64)

            self.episode += 1

        else:
            print("{} Test finished\n".format(self.path+self.name))
            print("Test: {}. Ticks: {}. Acum_Reward: {}. Avg_Reward: {}\n".format(self.episode, self.tick_episode, self.acum_reward, self.acum_reward/self.tick_episode))

            with open(self.path+self.name+"_test.txt", "a") as info:
                info.write("{},{},{},{}\n".format(self.episode, self.tick_episode, self.acum_reward, self.acum_reward/self.tick_episode))
                info.close()
                plt.plot()
            
            grafica = pd.read_csv(self.path+self.name+"_test.txt")

            plt.plot(grafica["ticks"])
            plt.savefig(self.path+self.name+"_ticks_test.png")
            plt.clf()
            plt.plot(grafica["acum_reward"])
            plt.savefig(self.path+self.name+"_reward_test.png")
            plt.clf()

            self.update_best_model()
            
        self.acum_loss = 0
        self.tick_episode = 0    
        self.acum_reward = 0
        self.acum_variance = 0
        self.save()

            #self.update_t_model()
            