import socket
import time
import random
import gym
import numpy as np
import pickle
import copy

import DeterministicAgent as da

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

    #Creacion del agente

    state_size = 6
    action_size = 7
    batch_size = 32

    agent = da.DeterministicAgent(state_size, action_size)

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
    s.settimeout(5)

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
        else:
            print("Connection established\n")

            try:
                data = conn.recv(SERVER_BUFFER_SIZE)
            except:
                print("No response (connection lost)\n")
                conn.close()
                print("Connection closed\n")
                continue
            else:

                decoded_data = decode_data(data)
                normalized_data = np.zeros(5)
                for idx, dat in enumerate(decoded_data):
                    normalized_data[idx] = (dat-infra_red_range[idx, 2])/(infra_red_range[idx, 1]-infra_red_range[idx, 2])

                previous_state = np.array([normalized_data[0], normalized_data[1], normalized_data[2], normalized_data[3], normalized_data[4], last_line])
                previous_state = np.reshape(previous_state, [1, state_size])

                previous_action = agent.act(normalized_data, normalized_limit)

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
                    elif line_position == 1 or line_position == 3:
                        reward = 10
                    elif line_position == 0  or line_position == 4:
                        reward = 5
                    last_line = line_position / 4
                    deactivateLT = 0
                else:
                    reward = 0
                    if deactivateLT > 50:
                        done = True
                    deactivateLT += 1

                state = np.array([normalized_data[0], normalized_data[1], normalized_data[2], normalized_data[3], normalized_data[4], last_line])
                state = np.reshape(state, [1, state_size])

                if not done:
                    action = agent.act(normalized_data, normalized_limit)
                else:
                    action = 8
                print(action)
                #action = 7
                conn.send(bytes(str(action),"ascii"))

                agent.remember(previous_state, previous_action, reward, state, done)

                previous_state = np.copy(state)
                previous_action = action

                if agent.tick % 100 == 0 and agent.tick > 0:
                    i=0
                    agent.save()
                
                agent.tick+=1
                agent.tick_episode+=1

                if done:
                    print("{} finished\n".format(agent.path+agent.name))
                    print("Episode: {}. Ticks: {}. Total ticks: {}\n".format(agent.episode, agent.tick_episode, agent.tick))

                    agent.save()
                    
                    agent.tick_episode = 0
                    last_line = 0.5
                    contador = 0
                    reward = 0
                    state = np.array([])
                    action = 7
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

                        decoded_data = decode_data(data)
                        normalized_data = np.zeros(5)
                        for idx, dat in enumerate(decoded_data):
                            normalized_data[idx] = (dat-infra_red_range[idx, 2])/(infra_red_range[idx, 1]-infra_red_range[idx, 2])

                        previous_state = np.array([normalized_data[0], normalized_data[1], normalized_data[2], normalized_data[3], normalized_data[4], last_line])
                        previous_state = np.reshape(previous_state, [1, state_size])

                        previous_action = agent.act(normalized_data, normalized_limit)

                        pck=bytes(str(previous_action),"ascii")

                        #print(pck)

                        conn.send(pck)

                #print("PC: ",time.time() - start)

