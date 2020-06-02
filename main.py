""" Main file. This is the starting point for your code execution.

You shouldn't need to change much of this code, but it's fine to as long as we
can still run your code with the arguments specified!
"""

import os
import json
import pickle
import argparse as ap

import numpy as np
import models
from data import load_image
import matplotlib

matplotlib.use('PS')
import matplotlib.pyplot as plt


def get_args():
    p = ap.ArgumentParser()

    # Meta arguments
    p.add_argument("--train-data", type=str, help="Training image file")
    p.add_argument("--model-file", type=str, required=True,
                   help="Where to store and load the model parameters")
    p.add_argument("--predictions-file", type=str,
                   help="Where to dump predictions")
    p.add_argument("--visualize-predictions-file", type=str,
                   help="Where to save visualization of segmentation")
    p.add_argument("--algorithm", type=str,
                   choices=['mrf'],
                   help="The type of model to use.")
    # Model Hyperparameters
    p.add_argument("--edge-weight", type=float, default=1.2,
                   help="MRF edge weight (J)")
    p.add_argument("--random-seed", type=int, default=1,
                   help="Random seed to ensure consistency among students")
    p.add_argument("--num-states", type=int, default=3,
                   help="The number of latent states (K)")
    p.add_argument("--n-em-iterations", type=int, default=10,
                   help="Number of EM iterations to run")
    p.add_argument("--n-vi-iterations", type=int, default=10,
                   help="Number of VI iterations to run")
    return p.parse_args()


# Visualize clustering
def visualize_clustering(image_output_file, predictions, X, K):
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(X, cmap='gray', vmin=0, vmax=255)
    plt.title('Raw data')
    plt.axis('off')
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(predictions)
    plt.title('Segmentation')
    plt.axis('off')
    plt.savefig(image_output_file)


def train(args):
    """ Fit a model's parameters given the parameters specified in args.
    """
    X = load_image(args.train_data)

    # Initialize appropriate algorithm
    if args.algorithm == 'mrf':
        model = models.MRF(J=args.edge_weight, K=args.num_states, n_em_iter=args.n_em_iterations,
                           n_vi_iter=args.n_vi_iterations)
    else:
        raise Exception("Algorithm argument not recognized")

    # Train the model
    model.fit(X=X)

    # Save the model
    pickle.dump(model, open(args.model_file, 'wb'))

    # predict most likely latent states for each of the pixels
    preds = model.predict(X)

    # output model predictions
    np.savetxt(args.predictions_file, preds, fmt='%s', delimiter='\t')

    # Visualize clustering
    visualize_clustering(args.visualize_predictions_file, preds, X, args.num_states)


if __name__ == "__main__":
    # Set seed
    ARGS = get_args()
    np.random.seed(ARGS.random_seed)
    train(ARGS)
