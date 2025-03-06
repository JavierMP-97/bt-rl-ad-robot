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
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import matplotlib.pyplot as plt
import math
import time
import pickle
import copy
from collections import deque

from keras import backend as K

import DDPGAgent as ddpg
import DQNAgent as dqn

np.set_printoptions(linewidth=np.inf)

'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
'''

print("\nTensorflow and keras loaded\n")

def decode_data(data):

    img = data
    #plt.imshow(data)
    #plt.show()

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = img[40:120,0:160]
    '''
    gray_img = skimage.color.rgb2gray(data)
    plt.imshow(gray_img)
    plt.show()
    print(gray_img)
    plt.imshow(gray_img2)
    plt.show()
    print(gray_img2)
    
    plt.imshow(gray_img)
    plt.show()
    '''
    restart = True
    for i in range(height):
        for j in range(width):
            if img[i,j] != img[0,0]:
                restart = False
                break
        if not restart:
            break

    #img = cv2.GaussianBlur(img, (3,3), 0)

    #plt.imshow(gray_img)
    #plt.show()

    img = cv2.Canny(img, 100, 250)
    img = img / 255

    #plt.imshow(img)
    #plt.show()
    
    decoded_data = img.reshape((height, width, channels))

    return decoded_data, restart

def decode_action(action):
    x = math.floor(action/5)
    y = action - x*5
    return [y * (2/4) -1, x * (2/2) -1]
    
print("Additional funcitons loaded\n")
    
if __name__ == '__main__':

    #Creacion del agente
    width = 160
    height = 80
    frame_stack = 3
    channels = 1

    state_size = (frame_stack, height, width, channels)
    ddpg_action_size = 2
    dqn_action_size = 15
    ddpg_action_range = 1
    batch_size = 32

    throttle = 0.3

    env = gym.make("donkey-generated-roads-v0")

    #agent = ddpg.DDPGAgent(ddpg_action_size, state_size, ddpg_action_range)

    agent = dqn.DQNAgent(state_size, dqn_action_size)

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

    train = True

    while agent.episode <= 300:

        state = np.zeros((1, frame_stack, height, width, channels))
        previous_state = np.zeros((1, frame_stack, height, width, channels))
        stack = np.zeros((1, 7, height, width, channels))
        action = np.zeros(2)
        done = False
        restart_counter = 0

        data = env.reset()    

        decoded_data,_ = decode_data(data["image"])
        
        '''
        for i in range(6):
            stack[0,i] = decoded_data

        state[0,0] = stack[0,0]
        state[0,1] = stack[0,3]
        state[0,2] = stack[0,6]
        '''

        s={}
        s["image"] = np.expand_dims(decoded_data, axis=0)
        s["speed"] = np.expand_dims(np.array([data["speed"]/30]), axis=0)
        previous_state = s.copy()

        restart_time = time.time()

        while not done:
            action = agent.act(s, train=train)
            
            if agent.tick_episode >= 2:
                decoded_action = decode_action(action)
            else:
                decoded_action = [0, 1]
                action = 12

            print(action)
            print(decoded_action)  
            data, reward, done, info = env.step(decoded_action) 
        
            #data, reward, done, info = env.step(action)
            if agent.tick_episode >= 500:
                done=True        
            decoded_data, restart = decode_data(data["image"])
            
            #plt.imshow(decoded_data)
            #plt.show()

            if restart:
                print("Simulator got stuck. Restarting.\n")

                #plt.imshow(data)
                #plt.show(block=False)

                env.render(close=True)

                time.sleep(1)

                env.render(close=False)
                break

            '''
            for i in range(6):
                stack[0,-i-1]=stack[0,-i-2]
            stack[0,0] = decoded_data

            
            state[0,0] = stack[0,0]
            state[0,1] = stack[0,3]
            state[0,2] = stack[0,6]
            '''

            s={}
            s["image"] = np.expand_dims(decoded_data, axis=0)
            s["speed"] = np.expand_dims(np.array([data["speed"]/30]), axis=0)

            if train:
                if not restart:
                    if agent.tick_episode >= 1:
                        #agent.remember(previous_state, action, reward, s, done)
                        agent.train((previous_state, action, reward, s, done))
                    
                    #if agent.tick >= 6:                  
                    #    agent.replay(agent.batch_size)

                    previous_state = s.copy()
            
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
            elif time.time() - restart_time > 180:
                data = env.reset()    

                decoded_data,_ = decode_data(data["image"])

                s={}
                s["image"] = np.expand_dims(decoded_data, axis=0)
                s["speed"] = np.expand_dims(np.array([data["speed"]/30]), axis=0)
                previous_state = s.copy()

                restart_time = time.time()
