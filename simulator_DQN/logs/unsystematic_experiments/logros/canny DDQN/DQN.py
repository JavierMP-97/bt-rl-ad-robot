import eventlet
eventlet.monkey_patch()

import logging
logging.getLogger('socketio').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

import sys

import gym
import donkey_gym
import random
import numpy as np
import cv2
import skimage as skimage
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Cropping3D, Conv3D, MaxPooling3D, BatchNormalization
import tensorflow as tf
from keras import backend as K
import matplotlib.pyplot as plt

import time
import pickle
import copy
from collections import deque
from keras.optimizers import Adam
from keras.layers.merge import Add

import tensorflow as tf
from keras import backend as K

import DQNAgent as dqn

np.set_printoptions(linewidth=np.inf)

'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
'''

print("\nTensorflow and keras loaded\n")

def decode_data(data):

    #plt.imshow(data)
    #plt.show()

    gray_img = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
    '''
    gray_img = skimage.color.rgb2gray(data)
    plt.imshow(gray_img)
    plt.show()
    print(gray_img)
    plt.imshow(gray_img2)
    plt.show()
    print(gray_img2)
    '''

    cannyed_img = cv2.Canny(gray_img, 100, 200) / 255
    #plt.imshow(cannyed_img)
    #plt.show()
    
    decoded_data = cannyed_img.reshape((height, width, 1))

    return decoded_data

def decode_action(action):
    return action * (2/6) -1
    
print("Additional funcitons loaded\n")
    
if __name__ == '__main__':

    #Creacion del agente
    width = 160
    height = 120
    frame_stack = 3 

    state_size = (frame_stack, height, width, 1)
    action_size = 7
    batch_size = 32

    throttle = 0.3

    env = gym.make("donkey-generated-roads-v0")

    agent = dqn.DQNAgent(state_size, action_size)

    #Carga del agente (o creacion de archivos si no existe)

    agent.load()

    #Pre_entrenamiento del agente
    '''
    with open( "DeterministicAgent.pr", "rb" ) as f:
        lista_tuplas = pickle.load(f)
        print(len(lista_tuplas["memory"]))
        agent.pre_train(lista_tuplas["memory"], 25)
    '''
    
    #Inicializacion del servidor

    agent.tick = 0
    agent.episode = 0

    train = True

    while 1:

        state = np.zeros((1, frame_stack, height, width, 1))
        previous_state = np.zeros((1, frame_stack, height, width, 1))
        action = 3
        previous_action = 3
        done = False

        data = env.reset()    

        decoded_data = decode_data(data)
        
        state[0,0] = decoded_data 
        state[0,1] = decoded_data 
        state[0,2] = decoded_data 

        previous_state = np.copy(state)

        while not done:
            
            action = agent.act(state, train=train)

            data, reward, done, info = env.step([decode_action(action), throttle])

            decoded_data = decode_data(data)

            restart = True
            for i in range(height):
                for j in range(width):
                    if decoded_data[i,j,0] != decoded_data[0,0,0]:
                        restart = False
                        break
                if not restart:
                    break
            
            if restart:
                print("Simulator got stuck. Restarting.\n")

                env.render(close=True)

                time.sleep(1)

                env.render(close=False)
                break

            state[0,2]=state[0,1]
            state[0,1]=state[0,0]
            state[0,0] = decoded_data

            if train:

                if agent.tick_episode >= 2:
                    #agent.remember(previous_state, action, reward, state, done)
                    agent.train((previous_state, action, reward, state, done))
                
                #if agent.tick >= 100:                  
                    #agent.replay(agent.batch_size)

                

                previous_state = np.copy(state)
            
                agent.tick += 1

            agent.tick_episode += 1
            agent.acum_reward += reward

            '''
            if agent.tick % 100 == 0:
                print(state)
                print(reward)
                print(done)
            '''

            #print(time.time()-start)
            if done:
                agent.next_episode(train=train)
                if train:
                    train = False
                else:
                    train = True
