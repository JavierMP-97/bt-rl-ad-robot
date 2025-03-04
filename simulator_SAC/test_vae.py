"""
Test a trained vae
"""
import argparse
import os

import cv2
import numpy as np
from stable_baselines.common import set_global_seeds

from vae.controller import VAEController
from config import ROI

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', help='Log folder', type=str, default='logs/recorded_data/')
parser.add_argument('-vae', '--vae-path', help='Path to saved VAE', type=str, default='')
parser.add_argument('--n-samples', help='Max number of samples', type=int, default=20)
parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
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

for i in range(args.n_samples):
    # Load test image
    #image_idx = np.random.randint(n_samples)
    #image_path = args.folder + images[image_idx]
    image_path = args.folder + images[i]
    image = cv2.imread(image_path)

    input_array[i] = np.copy(image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])])

    encoded = vae.encode_from_raw_image(image)
    reconstructed_image = vae.decode(encoded)[0]

    #print(reconstructed_image)

    output_array[i] = np.copy(reconstructed_image)

    #print(reconstructed_image.shape)
    # Plot reconstruction
    #cv2.imshow("Original", image)
    #cv2.imshow("Reconstruction", reconstructed_image)
    #cv2.waitKey(0)

'''
print("Input Array")
print(input_array)
print("\nOutput Array")
print(output_array)
print("\nSquare")
print(np.square(input_array - output_array))
print("\nSum")
'''

r_loss = np.square(input_array - output_array)
#r_loss = np.sum(r_loss, axis=(1, 2, 3))

#print(r_loss)
#print("\nMean")

r_loss = np.mean(r_loss)

print(r_loss)
