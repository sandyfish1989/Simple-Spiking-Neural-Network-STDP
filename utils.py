import tensorflow as tf
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio

def prepare_data(snn):
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
    X_train, Y_train = X_train[:snn.parameters.training_images_amount], Y_train[
                                                                         :snn.parameters.training_images_amount]
    X_test, Y_test = X_test[:snn.parameters.test_images_amount], Y_test[:snn.parameters.test_images_amount]

    return X_train, Y_train, X_test, Y_test


def save_checkpoint(epoch, synapses, neuron_labels_lookup, snn):

    weights_path = f"Checkpoints/{snn.readable_initial_timestamp}/Epoch_{epoch}/weights.csv"
    labels_path = f"Checkpoints/{snn.readable_initial_timestamp}/Epoch_{epoch}/labels.csv"
    visualized_weights_path = f"Visualized_Weights/{snn.readable_initial_timestamp}/Epoch_{epoch}/"

    # Use os.path.dirname to get the directory
    pathlib.Path(os.path.dirname(weights_path)).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.dirname(labels_path)).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.dirname(visualized_weights_path)).mkdir(parents=True, exist_ok=True)

    np.savetxt(weights_path, synapses, delimiter=",")
    np.savetxt(labels_path, neuron_labels_lookup, delimiter=',')



