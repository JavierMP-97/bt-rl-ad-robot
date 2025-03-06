"""
Test a trained vae
"""
import argparse
import os

import cv2
import numpy as np
import time

import numpy as np

from config import ROI, INPUT_DIM
import glob
import picamera
from picamera.array import PiYUVArray, PiRGBArray

from utils.tcp_utils import TcpClient

from utils.arduino_utils import ArduinoTransmitter
from threading import Event, Thread
from vae.controller import VAEController
import argparse

class CarCamera():
    def __init__(self):
        self.camera = picamera.PiCamera(resolution = (160, 120), framerate = 60, sensor_mode = 7)
        self.camera.start_preview()
        self.rawCapture = PiRGBArray(self.camera, size=(160,120))
        #self.image_array = np.zeros((120,160,3))
        self.image_array = np.zeros((80,160,3))
        #self.image_array = np.zeros((1,32))
        time.sleep(0.5)           
        #self.camera.stop_preview()
        #self.vae = VAEController()
        #self.vae.load("vae-level-0-dim-32.pkl")
        #print("Loaded VAE")
        self.start_process()
        time.sleep(0.5)
    
    def start_process(self):
        """Start main loop process."""
        self.process = Thread(target=self.main_loop)
        # Make it a deamon, so it will be deleted at the same time
        # of the main process
        self.process.daemon = True
        self.process.start()

    def main_loop(self):
        fotogram = 0
        per_sec = 0
        #im = 0
        while(1):
            
            if fotogram == 99:
                fotogram = 0
                print("FPS Camera: {}\n".format(100/(time.time()-per_sec)))

            if fotogram == 0:
                per_sec = time.time()

            fotogram += 1
            

            self.camera.capture(self.rawCapture, format="bgr", use_video_port = True)
            self.image_array = self.rawCapture.array[40:, 0:160]
            #cv2.imwrite("vae/train/im_{}.png".format(im), self.image_array)
            #im+=1
            #self.image_array = self.vae.encode(self.rawCapture.array[0:80, 0:160])
            self.rawCapture.truncate(0)

    def take_image(self):
        return self.image_array

class VAEProcessor:
    def __init__(self, vae_p):
        self.camera = CarCamera()
        print("Loaded camera")
        self.latent_vector = np.zeros((1,32))      
        self.vae = VAEController()
        self.vae.load(vae_p)
        self.start_process()
    
    def start_process(self):
        """Start main loop process."""
        self.process = Thread(target=self.main_loop)
        # Make it a deamon, so it will be deleted at the same time
        # of the main process
        self.process.daemon = True
        self.process.start()

    def main_loop(self):
        fotogram = 0
        per_sec = 0
        while(1):
            
            if fotogram == 99:
                fotogram = 0
                print("FPS VAE: {}\n".format(100/(time.time()-per_sec)))

            if fotogram == 0:
                per_sec = time.time()

            fotogram += 1
            
            #self.image_array = self.rawCapture.array
            self.latent_vector = self.vae.encode(self.camera.take_image())

    def take_image(self):
        return self.latent_vector

parser = argparse.ArgumentParser()
parser.add_argument('-vae', '--vae-path', help='Path to saved VAE', type=str, default='')
args = parser.parse_args()
if args.vae_path != '':
    vae_path = args.vae_path
else:
    vae_path = "vae-level-0-dim-32.pkl"
r = ROI

tcp_client = TcpClient("192.168.1.128", 33333)
tcp_client.open_connection()

print("Loaded tcp")

vae_proc = VAEProcessor(vae_path)

print("Loaded VAE {}".format(vae_path))

arduino_transmitter = ArduinoTransmitter()
arduino_transmitter.open_connection()

print("Loaded arduino")
out_of_bounds = bytes(10)
start = time.time()
while 1:

    #print("Starting loop\n")
    message_type = int(tcp_client.receive_message(1)[0])
    #print("Message type received: {}\n".format(message_type))
    if message_type == 1:

        action = tcp_client.receive_message(2)

        #steering = int(tcp_client.receive_message(1)[0]) / 128 - 1

        #print("Steering received: {}\n".format(steering))

        #throttle = int(tcp_client.receive_message(1)[0]) / 256

        #print("Throttle received: {}\n".format(throttle))
        #print("Steering received: {}\n".format(int(action[0])))
        #print("Throttle received: {}\n".format(int(action[1])))

        arduino_transmitter.send_message(action)
        out_of_bounds = arduino_transmitter.receive_data(11)

    elif message_type == 2:
        #t = time.time()
        t = time.time()

        image = vae_proc.take_image()
        #image = camera.vae.encode(image[0:80, 0:160])
        #image = np.delete(rawCapture.array, [1,2], 2)
        #time_cap = time.time() - t
        #t = time.time()
        #print(image.shape)

        #print("Image taken in {} sec\n".format(time.time()-t))

        #image_stream = cv2.imencode(".png", image, [int(cv2.IMWRITE_PNG_COMPRESSION), 6])[1].flatten() #, [int(cv2.IMWRITE_PNG_COMPRESSION), 3]

        #print(image_stream)

        #size = len(image_stream)

        #print(size)

        #image_stream = image.flatten()

        #encoded = vae.encode(image)
        
        '''
        for idx_h, height in enumerate(image):
            for idx_w, width in enumerate(height):
                for idx_c, channel in enumerate(width):
                   image_stream[idx_c + idx_w*INPUT_DIM[2] + idx_h*INPUT_DIM[1]] = image[idx_h, idx_w, idx_c]
        '''
        #print(len(image_stream))

        #image_stream = bytes(image_stream)
        image_stream = image.tobytes()

        image_stream += out_of_bounds

        #print("Image processed in {} sec\n".format(time.time()-t))

        time_proc = time.time() - t
        t = time.time()

        #tcp_client.send_message((size).to_bytes(4, byteorder='big'))
        tcp_client.send_message(image_stream)

        #print("Image sent in {} sec\n".format(time.time()-t))

        time_send = time.time() - t

        if time_proc + time_send > 0.075:
            print("Image processed in {} sec".format(time_proc))
            print("Data sent in {} sec".format(time_send))
            

    # Load test image
    #image_idx = np.random.randint(n_samples)
    #image_path = args.folder + images[image_idx]
    #image_path = args.folder + images[i]
    #image = cv2.imread(image_path)


    #input_array[i] = np.copy(image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])])
    #print("Image read")

    #print("Image encoded")
    #reconstructed_image = vae.decode(encoded)[0]

    #print(reconstructed_image)

    #output_array[i] = np.copy(reconstructed_image)

    #print(reconstructed_image.shape)
    # Plot reconstruction
    #cv2.imshow("Original", image)
    #cv2.imshow("Reconstruction", reconstructed_image)
    #cv2.waitKey(0)

#print("Captured {} frames at {}fps".format(n_samples, n_samples/(time.time() - start)))

'''
print("Input Array")
print(input_array)
print("\nOutput Array")
print(output_array)
print("\nSquare")
print(np.square(input_array - output_array))
print("\nSum")
'''

#r_loss = np.square(input_array - output_array)
#r_loss = np.sum(r_loss, axis=(1, 2, 3))

#print(r_loss)
#print("\nMean")

#r_loss = np.mean(r_loss)

#print(r_loss)