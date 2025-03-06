"""
Train a VAE model using saved images in a folder
"""
import argparse
import os

import cv2
import numpy as np
from stable_baselines.common import set_global_seeds
from tqdm import tqdm

from config import ROI
from vae.controller import VAEController
from vae.data_loader import DataLoader
from vae.model import ConvVAE
import tensorflow as tf

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', help='Path to a folder containing images for training', type=str,
                        default='logs\\recorded_data\\')
    parser.add_argument('--z-size', help='Latent space', type=int, default=512)
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument('--n-samples', help='Max number of samples', type=int, default=-1)
    parser.add_argument('--batch-size', help='Batch size', type=int, default=64)
    parser.add_argument('--learning-rate', help='Learning rate', type=float, default=1e-4)
    parser.add_argument('--kl-tolerance', help='KL tolerance (to cap KL loss)', type=float, default=0.5)
    parser.add_argument('--beta', help='Weight for kl loss', type=float, default=1.0)
    parser.add_argument('--n-epochs', help='Number of epochs', type=int, default=10)
    parser.add_argument('--verbose', help='Verbosity', type=int, default=1)
    parser.add_argument('--train', help='Train proportion', type=float, default=0.75)
    parser.add_argument('--test', help='Test', action='store_true', default=False)
    args = parser.parse_args()

    set_global_seeds(args.seed)

    if not args.folder.endswith('\\'):
        args.folder += '\\'

    vae = ConvVAE(z_size=args.z_size,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                kl_tolerance=args.kl_tolerance,
                beta=args.beta,
                is_training=True,
                reuse=False)

    #images = [im for im in os.listdir(args.folder) if im.endswith('.jpg')]
    images = [im for im in os.listdir(args.folder) if im.endswith('.png')]
    images = np.array(images)
    n_samples = len(images)

    if args.n_samples > 0:
        n_samples = min(n_samples, args.n_samples)

    print("{} images".format(n_samples))

    if args.train < 1:
        split_index = (int) (n_samples * args.train)

    # indices for all time steps where the episode continues
    indices = np.arange(n_samples, dtype='int64')
    np.random.shuffle(indices)

    if args.train < 1:
        train_indices = indices[:split_index]
        valid_indices = indices[split_index:]
        print(len(train_indices))
        print(len(valid_indices))
    else:
        train_indices = indices

    # split indices into minibatches. minibatchlist is a list of lists; each
    # list is the id of the observation preserved through the training
    minibatchlist_train = [np.array(sorted(train_indices[start_idx:start_idx + args.batch_size]))
                    for start_idx in range(0, len(train_indices) - args.batch_size + 1, args.batch_size)]

    if args.train < 1:    
        minibatchlist_valid = [np.array(sorted(valid_indices[start_idx:start_idx + args.batch_size]))
                        for start_idx in range(0, len(valid_indices) - args.batch_size + 1, args.batch_size)]


    data_loader_train = DataLoader(minibatchlist_train, images, n_workers=2, folder=args.folder)
    if args.train < 1:
        data_loader_valid = DataLoader(minibatchlist_valid, images, n_workers=2, folder=args.folder)

    if args.test:
        images_test = [im for im in os.listdir("logs\\recorded_data_test\\") if im.endswith('.png')]
        images_test = np.array(images_test)
        n_samples_test = len(images_test)
        test_indices = np.arange(n_samples_test, dtype='int64')
        np.random.shuffle(test_indices)
        minibatchlist_test = [np.array(sorted(test_indices[start_idx:start_idx + args.batch_size]))
                        for start_idx in range(0, len(test_indices) - args.batch_size + 1, args.batch_size)]
        data_loader_test = DataLoader(minibatchlist_test, images_test, n_workers=2, folder="logs\\recorded_data_test\\")

    vae_controller = VAEController(z_size=args.z_size)
    vae_controller.vae = vae

    for epoch in range(args.n_epochs):
        pbar = tqdm(total=len(minibatchlist_train))
        avg_train_loss = 0
        avg_r_loss = 0
        avg_kl_loss = 0
        for obs in data_loader_train:
            feed = {vae.input_tensor: obs}
            (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([
                vae.loss,
                vae.r_loss,
                vae.kl_loss,
                vae.global_step,
                vae.train_op
            ], feed)

            avg_train_loss += train_loss
            avg_r_loss += r_loss
            avg_kl_loss += kl_loss
            pbar.update(1)
        pbar.close()
        avg_train_loss = avg_train_loss/len(minibatchlist_train)
        avg_r_loss = avg_r_loss/len(minibatchlist_train)
        avg_kl_loss = avg_kl_loss/len(minibatchlist_train)
        print("Epoch {:3}/{}".format(epoch + 1, args.n_epochs))
        print("VAE: last optimization step", (train_step + 1), train_loss, r_loss, kl_loss)
        print("VAE: average optimization step", (train_step + 1), avg_train_loss, avg_r_loss, avg_kl_loss)

        # Update params
        vae_controller.set_target_params()

        if args.train < 1:
            pbar = tqdm(total=len(minibatchlist_valid))
            avg_test_loss = 0
            for obs in data_loader_valid:
                encoded = vae_controller.encode_batch(obs)
                reconstructed_image = vae_controller.decode_batch(encoded)
                test_r_loss = np.sum(
                        np.square(obs - reconstructed_image),
                        axis=(1, 2, 3)
                    )

                test_r_loss = np.mean(test_r_loss)
                avg_test_loss += test_r_loss
                pbar.update(1)
            pbar.close()

            avg_test_loss = avg_test_loss/len(minibatchlist_valid)

            print("VAE: validation loss", test_r_loss,"\n")

        if args.test:
            pbar = tqdm(total=len(minibatchlist_test))
            avg_test_loss = 0
            for obs in data_loader_test:
                encoded = vae_controller.encode_batch(obs)
                reconstructed_image = vae_controller.decode_batch(encoded)
                test_r_loss = np.sum(
                        np.square(obs - reconstructed_image),
                        axis=(1, 2, 3)
                    )

                test_r_loss = np.mean(test_r_loss)
                avg_test_loss += test_r_loss
                pbar.update(1)
            pbar.close()

            avg_test_loss = avg_test_loss/len(minibatchlist_test)

            print("VAE: test loss", test_r_loss,"\n")
        
        # Load test image
        if args.verbose >= 1:
            image_idx = np.random.randint(n_samples)
            image_path = args.folder + images[image_idx]
            image = cv2.imread(image_path)
            #r = ROI
            #im = image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
            im = image
            encoded = vae_controller.encode(im)
            reconstructed_image = vae_controller.decode(encoded)[0]
            # Plot reconstruction
            cv2.imshow("Original", image)
            cv2.imshow("Reconstruction", reconstructed_image)
            cv2.waitKey(1)

    save_path = "logs/vae-{}".format(args.z_size)
    os.makedirs(save_path, exist_ok=True)
    print("Saving to {}".format(save_path))
    vae_controller.set_target_params()
    vae_controller.save(save_path)
