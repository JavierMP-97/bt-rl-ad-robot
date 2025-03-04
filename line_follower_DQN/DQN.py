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

import DQNAgent as dqn

'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
'''

print("\nTensorflow and keras loaded\n")

def decode_data(data):

    #print(data)

    LT = np.zeros(5)

    #print(len(data))

    for i in range(5):
        LT[i] = data[i*2] << 8

        LT[i] += data[i*2+1]

    #print(data)

    #print(md)
    print(LT[0], LT[1], LT[2], LT[3], LT[4])

    return LT[0], LT[1], LT[2], LT[3], LT[4]
    
print("Additional funcitons loaded\n")
    
if __name__ == '__main__':
    
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

    state_size = 6
    action_size = 7
    batch_size = 32

    agent = dqn.DQNAgent(state_size, action_size)
    secondary_agent_number=""

    #Carga del agente (o creacion de archivos si no existe)

    agent.load()

    #Pre_entrenamiento del agente
    '''
    with open( "DeterministicAgent.pr", "rb" ) as f:
        lista_tuplas = pickle.load(f)
        print(len(lista_tuplas["memory"]))
        agent.pre_train(lista_tuplas["memory"], 25)
    '''
    #Carga del calibrado

    with open( "calibration.pr", "rb" ) as f:
        infra_red_range = pickle.load(f)

    normalized_limit = np.zeros(5)

    for idx, irr in enumerate(infra_red_range):
        normalized_limit[idx] = (((irr[0] + irr[2]) / 2) - irr[2]) / (irr[1]-irr[2])
    
    #Inicializacion del servidor

    TCP_IP = ''
    TCP_PORT = 5000
    SERVER_BUFFER_SIZE = 10

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_IP, TCP_PORT))
    s.listen(1)
    s.settimeout(3)

    print("Created server on port: ",TCP_PORT,"\n")

    agent.tick = 0
    agent.episode = 0

    while 1:
        #deactivateUS=0      #Contador para parar el vehiculo (ultrasonidos)
        deactivateLT = 0      #Contador para parar el vehiculo (line tracker)
        agent.tick_episode = 0
        last_line = 0.5
        contador = 0
        reward = 0
        state = np.array([])
        previous_state = np.array([])
        action = 7
        previous_action = 3
        done = False

        try:
            print("Waiting for connection...\n")
            conn, addr = s.accept()
        except:
            print("A connection couldn't be established. Trying again.\n")
            continue
        
        print("Connection established\n")

        try:
            data = conn.recv(SERVER_BUFFER_SIZE)
        except:
            print("No response (connection lost)\n")
            conn.close()
            print("Connection closed\n")
            continue
        

        decoded_data = decode_data(data)
        normalized_data = np.zeros(5)
        for idx, dat in enumerate(decoded_data):
            normalized_data[idx] = (dat-infra_red_range[idx, 2])/(infra_red_range[idx, 1]-infra_red_range[idx, 2])

        previous_state = np.array([normalized_data[0], normalized_data[1], normalized_data[2], normalized_data[3], normalized_data[4], last_line])
        previous_state = np.reshape(previous_state, [1, state_size])

        previous_action = agent.act(previous_state)

        pck=bytes(str(previous_action),"ascii")

        #print(pck)

        conn.send(pck)

        while 1:
            #start = time.time()

            try:
                data = conn.recv(SERVER_BUFFER_SIZE)
            except:
                print("No response (connection lost)\n")
                conn.close()
                print("Connection closed\n")
                break

            #print("Arduino: ",time.time() - start)

            #start = time.time()

            decoded_data = decode_data(data)
            normalized_data = np.zeros(5)
            for idx, dat in enumerate(decoded_data):
                normalized_data[idx] = (dat-infra_red_range[idx, 2])/(infra_red_range[idx, 1]-infra_red_range[idx, 2])
            line_position = np.argmin(normalized_data)

            if normalized_data[line_position] <= (normalized_limit[line_position] * 1.5):
                if line_position == 2:
                    reward = 15
                elif (line_position == 1 or line_position == 3):
                    reward = 10
                elif (line_position == 0  or line_position == 4):
                    reward = 5
                #last_line = line_position / 4
                if (line_position == 0 or line_position == 1):
                    last_line = 0
                elif (line_position == 3 or line_position == 4):
                    last_line = 0.5
                else:
                    last_line = 1
                deactivateLT = 0
            else:
                reward = 0
                if deactivateLT > 10:
                    done = True
                deactivateLT += 1

            state = np.array([normalized_data[0], normalized_data[1], normalized_data[2], normalized_data[3], normalized_data[4], last_line])
            state = np.reshape(state, [1, state_size])

            if not done:
                action = agent.act(state)
            else:
                action = 7
            print(action)
            #action = 7
            conn.send(bytes(str(action),"ascii"))

            #agent.train([previous_state, previous_action, reward, state, done])

            agent.remember(previous_state, previous_action, reward, state, done)

            agent.replay(agent.batch_size)

            previous_state = np.copy(state)
            previous_action = action

            if agent.tick % 1000 == 0 and agent.tick > 0:
                i=0
                agent.save()
            
            agent.tick+=1
            agent.tick_episode+=1

            if done:
                print("{} finished\n".format(agent.path+agent.name))
                print("Episode: {}. Ticks: {}. Epsilon: {}. Total ticks: {}\n".format(agent.episode, agent.tick_episode, agent.epsilon, agent.tick))

                with open(agent.path+agent.name+".txt", "a") as info:
                    info.write("Episode: {}. Ticks: {}. Epsilon: {}. Total ticks: {}\n".format(agent.episode, agent.tick_episode, agent.epsilon, agent.tick))
                    info.close()

                print("Replay started...\n")
                #for q in range(1):
                #    agent.replay(64)
                
                agent.save()

                print("Returning to line\n")
                not_returned = True
                while not_returned:

                    try:
                        print("hola")
                        data = conn.recv(SERVER_BUFFER_SIZE)
                    except:
                        print("No response (connection lost)\n")
                        conn.close()
                        print("Connection closed\n")
                        break
                    decoded_data = decode_data(data)
                    normalized_data = np.zeros(5)
                    for idx, dat in enumerate(decoded_data):
                        normalized_data[idx] = (dat-infra_red_range[idx, 2])/(infra_red_range[idx, 1]-infra_red_range[idx, 2])
                    line_position = np.argmin(normalized_data)

                    if normalized_data[line_position] <= (normalized_limit[line_position] * 1.5):
                        action = 7
                        print(action)
                        conn.send(bytes(str(action),"ascii"))
                        time.sleep(1)
                        not_returned = False
                        print("Agent returned\n")
                    else:
                        if last_line < 0.5:
                            action = 0
                            print(action)
                            conn.send(bytes(str(action),"ascii"))
                        elif last_line > 0.5:
                            action = 6
                            print(action)
                            conn.send(bytes(str(action),"ascii"))
                
                agent.tick_episode = 0
                last_line = 0.5
                contador = 0
                reward = 0
                state = np.array([])
                action = 7
                agent.episode+=1
                done=False
                deactivateLT=0



                try:
                    data = conn.recv(SERVER_BUFFER_SIZE)
                except:
                    print("No response (connection lost)\n")
                    conn.close()
                    print("Connection closed\n")
                    break
                

                decoded_data = decode_data(data)
                normalized_data = np.zeros(5)
                for idx, dat in enumerate(decoded_data):
                    normalized_data[idx] = (dat-infra_red_range[idx, 2])/(infra_red_range[idx, 1]-infra_red_range[idx, 2])

                previous_state = np.array([normalized_data[0], normalized_data[1], normalized_data[2], normalized_data[3], normalized_data[4], last_line])
                previous_state = np.reshape(previous_state, [1, state_size])

                previous_action = agent.act(previous_state)

                pck=bytes(str(previous_action),"ascii")

                conn.send(pck)

