"""
Test a trained vae
"""
import argparse
import os

import cv2
import numpy as np
from stable_baselines.common import set_global_seeds
import time

from vae.controller import VAEController
from config import ROI

from config import ENV_ID
from utils.utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
import glob
import picamera
from picamera.array import PiYUVArray, PiRGBArray


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', help='Log folder', type=str, default='logs/recorded_data/')
parser.add_argument('-vae', '--vae-path', help='Path to saved VAE', type=str, default='')
parser.add_argument('--n-samples', help='Max number of samples', type=int, default=20)
parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
parser.add_argument('--algo', help='RL Algorithm', default='sac',
                    type=str, required=False, choices=list(ALGOS.keys()))
parser.add_argument('--exp-id', help='Experiment ID (-1: no exp folder, 0: latest)', default=0,
                    type=int)
parser.add_argument('-best', '--best-model', action='store_true', default=True,
                    help='Use best saved model of that experiment (if it exists)')
parser.add_argument('--mfolder', help='Log folder model', type=str, default='logs')
args = parser.parse_args()

set_global_seeds(args.seed)

if not args.folder.endswith('/'):
    args.folder += '/'

vae = VAEController()
vae.load(args.vae_path)

images = [im for im in os.listdir(args.folder) if im.endswith('.jpg')]
images = np.array(images)
n_samples = len(images)

n_samples = min(n_samples, args.n_samples)

r = ROI

input_array = np.zeros((n_samples,r[3],r[2],3))
output_array = np.zeros((n_samples,r[3],r[2],3))

print("Loaded names")

algo = args.algo
folder = args.mfolder

if args.exp_id == 0:
    args.exp_id = get_latest_run_id(os.path.join(folder, algo), ENV_ID)
    print('Loading latest experiment, id={}'.format(args.exp_id))

# Sanity checks
if args.exp_id > 0:
    log_path = os.path.join(folder, algo, '{}_{}'.format(ENV_ID, args.exp_id))
else:
    log_path = os.path.join(folder, algo)

best_path = ''
if args.best_model:
    best_path = '_best'

model_path = os.path.join(log_path, "{}{}.pkl".format(ENV_ID, best_path))

assert os.path.isdir(log_path), "The {} folder was not found".format(log_path)
assert os.path.isfile(model_path), "No model found for {} on {}, path: {}".format(algo, ENV_ID, model_path)


model = ALGOS[algo].load(model_path)

command_history = np.zeros((1, 4))

print("Loaded model")

camera = picamera.PiCamera()
camera.resolution = (160, 120)
camera.framerate = 90
camera.sensor_mode = 7
camera.start_preview()
rawCapture = PiRGBArray(camera, size=(160,120))
time.sleep(3)
camera.stop_preview()
print("Loaded camera")

start = time.time()
for i in range(n_samples):
    # Load test image
    #image_idx = np.random.randint(n_samples)
    #image_path = args.folder + images[image_idx]
    #image_path = args.folder + images[i]
    #image = cv2.imread(image_path)
    camera.capture(rawCapture, format="rgb", use_video_port = True)
    #image = np.delete(rawCapture.array, [1,2], 2)
    image = rawCapture.array
    #print(image.shape)
    image = image[0:80, 0:160]

    #input_array[i] = np.copy(image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])])
    #print("Image read")
    encoded = vae.encode(image)
    observation = np.concatenate((encoded, command_history), axis=-1)
    _, _ = model.predict(observation, deterministic=True)
    rawCapture.truncate(0)
    #print("Image encoded")
    #reconstructed_image = vae.decode(encoded)[0]

    #print(reconstructed_image)

    #output_array[i] = np.copy(reconstructed_image)

    #print(reconstructed_image.shape)
    # Plot reconstruction
    #cv2.imshow("Original", image)
    #cv2.imshow("Reconstruction", reconstructed_image)
    #cv2.waitKey(0)

print("Captured {} frames at {}fps".format(n_samples, n_samples/(time.time() - start)))

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
