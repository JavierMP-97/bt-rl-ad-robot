import socket
import time
import random
import gym
import numpy as np
import pickle
from collections import deque
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Lambda
from keras.optimizers import Adam
from keras.layers.merge import Add

import tensorflow as tf
from keras import backend as K

'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
'''

print("\nTensorflow and keras loaded\n")

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
    def __init__(self, state_size, action_size, name="LT_Agent", path="./save/", memory_buffer=16384,   \
    gamma=0.95, epsilon= 0.7, epsilon_min=0.01, epsilon_decay=0.9995,     \
    learning_rate=0.001, learning_rate_decay=0.0, tau_max=4,        \
    batch_size=16, upgrade_mode=0, model_type=0):
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
            model.add(Dense(20, input_dim=self.state_size, activation='tanh'))
            model.add(Dense(40, activation='tanh'))
            model.add(Dense(self.action_size, activation='linear')) 

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
        self.memory.append((state, action, reward, next_state, done))

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

    def update_t_model(self):
        self.t_model.set_weights(self.model.get_weights())

    def load(self):
        try:
            d = pickle.load( open( self.path+self.name+".pr", "rb" ) )
            self.epsilon = d["epsilon"]
            #self.epsilon = 0.3
            self.memory.extend(d["memory"])
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
        d["memory"] = self.memory
        d["tau"] = self.tau
        pickle.dump( d, open( self.path+self.name+".pr" , "wb" ), protocol=pickle.HIGHEST_PROTOCOL )
        self.model.save_weights(self.path+self.name+".hdf5")

        print("{} saved\n".format(self.path+self.name))
        #self.t_model.save_weights(self.path+self.name+"_t.hdf5")

def decode_data(data):

    md = data[0]/200

    LT=data[1]-1

    LT_L = (LT & 4)>>2
    LT_M = (LT & 2)>>1
    LT_R = (LT & 1)

    #print(data)

    #print(md)
    print(LT_L,LT_M,LT_R)

    return md,LT_L,LT_M,LT_R
    
print("Additional funcitons loaded\n")
    
if __name__ == '__main__':

    AGENT_PATH = "./save/LT_Agent"
    
    #Usar/no usar GPU

    num_cores = 4

    GPU=False

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

    #Creacion del agente

    state_size = 4
    action_size = 6
    batch_size = 32

    agent = DQNAgent(state_size, action_size)
    secondary_agent_number=""

    #Carga del agente (o creacion de archivos si no existe)

    agent.load()
    
    #Inicializacion del servidor

    TCP_IP = ''
    TCP_PORT = 5000
    SERVER_BUFFER_SIZE = 2

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_IP, TCP_PORT))
    s.listen(1)
    s.settimeout(20)

    print("Created server on port: ",TCP_PORT,"\n")

    agent.tick=0
    agent.episode=0

    while 1:
        #deactivateUS=0      #Contador para parar el vehiculo (ultrasonidos)
        deactivateLT=0      #Contador para parar el vehiculo (line tracker)
        agent.tick_episode=0
        ultimo_LT=1
        contador = 0
        reward = 0
        state = np.array([])
        action = 5
        done=False

        print("Waiting for connection...\n")

        try:
            conn, addr = s.accept()
        except:
            print("A connection couldn't be established. Trying again.\n")
        else:
            print("Connection established\n")

            try:
                data = conn.recv(SERVER_BUFFER_SIZE)
            except:
                print("No response (connection lost)\n")
                conn.close()
                print("Connection closed\n")
            else:

                md,LT_L,LT_M,LT_R = decode_data(data)

                if LT_L==0 and LT_M==0 and LT_R==0:
                    contador+=1
                    
                if LT_L:
                    ultimo_LT = 0
                elif LT_R:
                    ultimo_LT = 2
                elif LT_M:
                    ultimo_LT = 1

                previous_state = np.array([LT_L,LT_M,LT_R,ultimo_LT])
                previous_state = np.reshape(previous_state, [1, state_size])

                previous_action = 5

                pck=bytes(str(previous_action),"ascii")

                #print(pck)

                conn.send(pck)

                while 1:
                    #start = time.time()

                    try:
                        data = conn.recv(SERVER_BUFFER_SIZE)
                    except:
                        print("No response (connection lost)\n")
                        #print("Replay started...\n")
                        #agent.replay(agent.batch_size)
                        #agent.save(AGENT_PATH)
                        #print("{} saved\n".format(AGENT_PATH))
                        conn.close()
                        print("Connection closed\n")
                        break

                    #print("Arduino: ",time.time() - start)

                    #start = time.time()

                    md,LT_L,LT_M,LT_R = decode_data(data)

                    #if md<0.075:
                        #reward-=100
                        #deactivateUS+=1

                    if previous_action == 1 or previous_action == 2 or previous_action == 3:
                        reward = 2
                    elif previous_action == 0 or previous_action == 4:
                        reward = 1
                    else:
                        reward = 0
                        
                    if LT_L==0 and LT_M==0 and LT_R==0:
                        #deactivateUS=0
                        if deactivateLT>=25:
                            reward-=10
                            done = True
                        elif contador>=25:
                            reward-=10
                            deactivateLT+=1
                        else:
                            contador+=1
                    else:
                        deactivateLT=0
                        deactivateUS=0
                        contador=0
                    
                    if LT_L:
                        ultimo_LT = 0
                    elif LT_R:
                        ultimo_LT = 2
                    elif LT_M:
                        ultimo_LT = 1

                    state = np.array([LT_L,LT_M,LT_R,ultimo_LT])
                    state = np.reshape(state, [1, state_size])

                    if not done:
                        action = agent.act(state)
                    else:
                        action = 6

                    conn.send(bytes(str(action),"ascii"))

                    agent.train([previous_state, previous_action, reward, state, done])

                    agent.remember(previous_state, previous_action, reward, state, done)

                    agent.replay(agent.batch_size)

                    previous_state = state
                    previous_action = action

                    if agent.tick == 1000:
                        i=0
                        agent.save()
                    
                    agent.tick+=1
                    agent.tick_episode+=1

                    if done:
                        print("{} finished\n".format(AGENT_PATH))
                        print("Episode: {}. Ticks: {}. Epsilon: {}. Total ticks: {}\n".format(agent.episode, agent.tick_episode, agent.epsilon, agent.tick))

                        info = open(AGENT_PATH+".txt", "a")
                        info.write("Episode: {}. Ticks: {}. Epsilon: {}. Total ticks: {}\n".format(agent.episode, agent.tick_episode, agent.epsilon, agent.tick))
                        info.close()

                        print("Replay started...\n")
                        for q in range(1):
                            agent.replay(64)
                        
                        agent.save()
                        
                        agent.tick_episode=0
                        ultimo_LT=1
                        contador = 0
                        reward = 0
                        state = np.array([])
                        action = 5
                        agent.episode+=1
                        done=False
                        deactivateLT=0

                        #conn.close()
                        #break
                        try:
                            data = conn.recv(SERVER_BUFFER_SIZE)
                        except:
                            print("No response (connection lost)\n")
                            conn.close()
                            print("Connection closed\n")
                            break
                        else:

                            md,LT_L,LT_M,LT_R = decode_data(data)

                            if LT_L==0 and LT_M==0 and LT_R==0:
                                contador+=1
                                
                            if LT_L:
                                ultimo_LT = 0
                            elif LT_R:
                                ultimo_LT = 2
                            elif LT_M:
                                ultimo_LT = 1

                            previous_state = np.array([LT_L,LT_M,LT_R,ultimo_LT])
                            previous_state = np.reshape(previous_state, [1, state_size])

                            previous_action = 5

                            pck=bytes(str(previous_action),"ascii")

                            #print(pck)

                            conn.send(pck)

                    #print("PC: ",time.time() - start)

