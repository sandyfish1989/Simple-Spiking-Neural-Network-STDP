import tensorflow as tf
import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
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

    if snn.parameters.visualize_weights:
        visualize_synapse_weights_and_save(neuron_labels_lookup, synapses, visualized_weights_path, snn)


def visualize_synapse_weights_and_save(label_neuron, synapses, save_path, snn):
    for layer2_index in range(snn.parameters.layer2_size):
        if label_neuron[layer2_index] == -1:
            for layer1_index in range(snn.parameters.layer1_size):
                synapses[layer2_index][layer1_index] = 0
        image = convert_weights_to_image(synapses[layer2_index])
        imageio.imwrite(save_path / f'Neuron_{layer2_index}.png', image.astype(np.uint8))


def convert_weights_to_image(self, weights):
    weights = np.array(weights)
    weights = np.reshape(weights, self.parameters.image_size)
    image = np.zeros(self.parameters.image_size)
    for x_coordinate in range(self.parameters.image_size[0]):
        for y_coordinate in range(self.parameters.image_size[1]):
            image[x_coordinate][y_coordinate] = int(
                np.interp(weights[x_coordinate][y_coordinate], [self.parameters.min_weight, self.parameters.max_weight], [0, 255]))
    return image

def plot_potentials_over_time(self, potential_thresholds, potentials):
    # Plotting
    spaced_potentials = np.arange(0, len(potentials[0]), 1)
    for i in range(self.parameters.layer2_size):
        axes = plt.gca()
        plt.plot(spaced_potentials, potential_thresholds[i], 'r')
        plt.plot(spaced_potentials, potentials[i])
        plt.show()