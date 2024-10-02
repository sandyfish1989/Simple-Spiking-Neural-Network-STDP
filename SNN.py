import time
import imageio.v3 as iio
import numpy as np
import math
import utils

from Neuron import Neuron
from Parameters import Parameters


class SNN:
    def __init__(self, parameters: Parameters):
        self.parameters = parameters

        self.initial_timestamp = time.time()
        self.readable_initial_timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")


    def encode_image_to_spike_train(self, image: np.ndarray):
        spike_trains = []

        for x_position in range(image.shape[0]):
            for y_position in range(image.shape[1]):

                pixel_value = image[x_position][y_position]

                spike_train = np.zeros(shape=(self.parameters.image_train_time + 1,))

                # Transfer pixel value to set frequency range(and some other stuff, which interp does...)
                frequency = np.interp(pixel_value, [np.min(image), np.max(image)], [self.parameters.min_frequency, self.parameters.max_frequency])

                spike_time_distance = math.ceil(self.parameters.image_train_time / frequency)
                next_spike_time = spike_time_distance

                if pixel_value > 0:
                    while next_spike_time < (self.parameters.image_train_time + 1):
                        # Add Spike to Spike Train
                        spike_train[int(next_spike_time)] = 1

                        # Calculate next spike
                        next_spike_time += spike_time_distance

                spike_trains.append(spike_train)

        return spike_trains  # (784, times)

    def grey_image(self, image: np.ndarray):
        return image / 255.0


    # STDP reinforcement learning curve
    def STDP_weighting_curve(self, delta_time: int):
        if delta_time < 0: # LTP
            mapped_ltp_time = 400 + (1600 - 400) * (-delta_time) / 20
            A_plus = 2.48406  # Amplitude of LTP (Long-Term Potentiation) when Δt > 0
            tau_plus = 483.6014  # Time constant for LTP
            y0_plus = -0.0654
            delta_W = A_plus * np.exp(-mapped_ltp_time / tau_plus) + y0_plus
            return delta_W * self.parameters.ltp_lr
        else:
            mapped_ltd_time = -200 - (1600 - 200) * delta_time / 20
            A_minus = -1.54537  # Amplitude of LTD (Long-Term Depression) when Δt < 0
            tau_minus = -393.8205  # Time constant for LTD
            y0_minus = -0.15
            delta_W = A_minus * np.exp(-mapped_ltd_time / tau_minus) + y0_minus
            return delta_W * self.parameters.ltd_lr


    # STDP weight update rule
    def update_synapse(self, synapse_weight, weight_factor):
        if weight_factor < 0:
            return synapse_weight + self.parameters.sigma * weight_factor * (synapse_weight - abs(self.parameters.min_weight)) ** self.parameters.mu
        elif weight_factor > 0:
            return synapse_weight + self.parameters.sigma * weight_factor * (self.parameters.max_weight - synapse_weight) ** self.parameters.mu


    def train(self):
        print("Starting Training...")

        # Update initial timestamp
        self.initial_timestamp = time.time()
        self.readable_initial_timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")

        testing_accuracies = []

        time_of_learning = np.arange(1, self.parameters.image_train_time + 1, 1)

        output_layer = [Neuron(self.parameters) for i in range(self.parameters.layer2_size)]

        # Random Synapse Matrix	Initialization
        synapses = np.ones((self.parameters.layer2_size,
                            self.parameters.layer1_size))  # Alternative: np.random.uniform(low=0.95, high=1.0, size=(self.parameters.layer2_size, self.parameters.layer1_size))  # Working Option: np.ones((layer2_size , layer1_size))
        self.parameters.max_weight = np.max(synapses)

        synapse_memory = np.zeros((self.parameters.layer2_size, self.parameters.layer1_size))

        # Creating Mapping Neurons which contains the Number they have learned
        neuron_labels_lookup = np.zeros((self.parameters.layer2_size, 10))

        X_train, Y_train, X_test, Y_test = utils.prepare_data(self)

        # Starting training
        for epoch in range(self.parameters.epochs):

            # Iterating over each image and coresponding label
            for image, label in zip(X_train, Y_train):
                time_start = time.time()

                # Convolving image with receptive field and encoding to generate spike train
                spike_train = np.array(self.encode_image_to_spike_train(self.grey_image(image)))

                # Local variables
                winner_index = None
                count_spikes = np.zeros(self.parameters.layer2_size)
                current_potentials = np.zeros(self.parameters.layer2_size)

                synapse_memory = np.zeros((self.parameters.layer2_size, self.parameters.layer1_size))

                # Leaky integrate and fire neuron dynamics
                for time_step in time_of_learning:

                    for neuron_index, neuron in enumerate(output_layer):
                        self.calculate_potentials_and_adapt_thresholds(current_potentials, neuron, neuron_index, spike_train, synapses, time_step)

                    # Determine the winner neuron and index
                    winner_index, winner_neuron = self.get_winner_neuron(current_potentials, output_layer)

                    # Check if Winner doesn't spike(reaches its own adaptive spike threshold?)
                    if current_potentials[winner_index] < winner_neuron.adaptive_spike_threshold:
                        continue  # Go to next time step

                    count_spikes[winner_index] += 1

                    winner_neuron.hyperpolarization(time_step)
                    winner_neuron.adaptive_spike_threshold += 1  # Adaptive Membrane / Homoeostasis: Increasing the threshold of the neuron

                    for layer1_index in range(self.parameters.layer1_size):
                        self.find_and_strengthen_contributing_synapses(layer1_index, spike_train, synapses, time_step, winner_index, synapse_memory)
                        self.find_and_weaken_not_contributing_synapses(layer1_index, synapse_memory, synapses, winner_index)

                    self.inihibit_looser_neurons(count_spikes, output_layer, time_step, winner_index)

                # Reset the neurons for the next image
                self.reset_neurons(output_layer)

                # Assigning Label to Winner Neuron
                neuron_labels_lookup[winner_index][int(label)] += 1

            # TODO PROBABLY WRONG AND DANGEROUS OR SUPER IMPORTANT
            """for layer2_index in range(self.parameters.layer2_size):
                if neuron_labels_lookup[layer2_index] == -1:
                    for layer1_index in range(self.parameters.layer1_size):
                        synapses[layer2_index][layer1_index] = 0
"""         
            # Accuracy tested against test dataset after each epoch
            if self.parameters.testing:
                testing_accuracy = self.test(synapses, np.argmax(neuron_labels_lookup, axis=1), (X_test, Y_test))
                testing_accuracies.append(testing_accuracy)

            if epoch % self.parameters.debugging_interval == 0:
                self.debug_training_state(epoch, count_spikes, testing_accuracies)

            if epoch % self.parameters.checkpoint_interval == 0 and self.parameters.checkpoint_interval != 0:
                utils.save_checkpoint(epoch, synapses, np.argmax(neuron_labels_lookup, axis=1), self)

        # Final Checkpoint
        utils.save_checkpoint("Final", synapses, np.argmax(neuron_labels_lookup, axis=1), self)

        print("Finished Training. Saved Weights and Labels.")


    # Tests the SNN for its accuracy
    def test(self, synapses, neuron_labels_lookup, dataset):
        X_test, Y_test = dataset
        predictions = []
        actual_labels = []
        for image, label in zip(X_test, Y_test):
            spike_train = np.array(self.encode_image_to_spike_train(self.grey_image(image)))
            prediction = self.infer(spike_train, synapses, neuron_labels_lookup)
            predictions.append(prediction)
            actual_labels.append(label)

        # Calculate Accuracy
        correct_predictions = 0
        for i in range(len(predictions)):
            if int(predictions[i]) == int(actual_labels[i]):
                correct_predictions += 1
        accuracy = correct_predictions / len(predictions)

        return accuracy

    # Used to classify one image
    def inference(self, image_path, synapse_weights_path=None, labels_matrix_path=None):
        if synapse_weights_path is None:
            synapse_weights_path = self.parameters.weights_path
        if labels_matrix_path is None:
            labels_matrix_path = self.parameters.labels_path
        image = iio.imread(image_path)
        spike_train = np.array(self.encode_image_to_spike_train(self.grey_image(image)))
        synapses = np.loadtxt(synapse_weights_path, delimiter=",")
        neuron_labels_lookup = np.loadtxt(labels_matrix_path, delimiter=',')

        prediction = self.infer(spike_train, synapses, neuron_labels_lookup)

        return prediction

    def infer(self, spike_train, synapses, neuron_labels_lookup):
        synapses = synapses.copy()
        neuron_labels_lookup = neuron_labels_lookup.copy()

        # time series
        time_of_learning = np.arange(1, self.parameters.image_train_time + 1, 1)
        count_spikes = np.zeros((self.parameters.layer2_size, 1))
        output_layer = [Neuron(self.parameters) for i in range(self.parameters.layer2_size)]
        # flag for lateral inhibition
        current_potentials = np.zeros(self.parameters.layer2_size)
        winner_index = None
        for time_step in time_of_learning:
            for layer2_index, layer2_neuron in enumerate(output_layer):
                self.calculate_potentials_and_adapt_thresholds(current_potentials, layer2_neuron, layer2_index, spike_train, synapses, time_step)

            winner_index, winner_neuron = self.get_winner_neuron(current_potentials, output_layer)

            # Check if Winner doesn't spike(reaches its own adaptive spike threshold?)
            if current_potentials[winner_index] < winner_neuron.adaptive_spike_threshold:
                continue  # Go to next time step

            count_spikes[winner_index] += 1
            winner_neuron.hyperpolarization(time_step)
            winner_neuron.adaptive_spike_threshold += 1  # Adaptive Membrane / Homoeostasis: Increasing the threshold of the neuron #TODO PLUS OR MINUS?!?

            self.inihibit_looser_neurons(count_spikes, output_layer, time_step, winner_index)
        prediction = neuron_labels_lookup[np.argmax(count_spikes)]
        return prediction


    def debug_training_state(self, epoch, count_spikes, testing_accuracies):
        print(f"Epoch: {epoch} Testing Accuracy: {round(testing_accuracies[-1], 2) if testing_accuracies else 'None'}"
              f" Time passed: {round(time.time() - self.initial_timestamp, 2)} seconds")

        # To write intermediate synapses for neurons
        # for p in range(layer2_size):
        #	reconst_weights(synapse[p],str(p)+"_epoch_"+str(k))

    def reset_neurons(self, output_layer):
        # Bring neuron potentials to rest
        for neuron_index, neuron in enumerate(output_layer):
            neuron.initial()

    def inihibit_looser_neurons(self, count_spikes, output_layer, time_step, winner_index):
        # Inhibit all LOOSERS
        for looser_neuron_index, looser_neuron in enumerate(output_layer):
            if looser_neuron_index != winner_index:
                if looser_neuron.potential > looser_neuron.adaptive_spike_threshold:
                    count_spikes[looser_neuron_index] += 1

                looser_neuron.inhibit(time_step)

    def find_and_strengthen_contributing_synapses(self, layer1_index, spike_train, synapses, time_step, winner_index, synapse_memory):
        """Part of STDP - Any synapse that contribute to the firing of a post-synaptic neuron should be increased. Depending on the timing of the pre- and postsynaptic spikes."""
        for past_time_step in range(0, self.parameters.past_window - 1, -1):  # if presynaptic spike came before postsynaptic spike
            if 0 <= time_step + past_time_step < self.parameters.image_train_time + 1:
                if spike_train[layer1_index][time_step + past_time_step] == 1:  # if presynaptic spike was in the tolerance window
                    synapses[winner_index][layer1_index] = self.update_synapse(synapses[winner_index][layer1_index], self.STDP_weighting_curve(past_time_step))  # strengthen weights
                    synapse_memory[winner_index][layer1_index] = 1
                    break
                # else:
                # synapse_memory[winner_index][layer1_index] = 0

    def find_and_weaken_not_contributing_synapses(self, layer1_index, synapse_memory, synapses, winner_index):
        if synapse_memory[winner_index][layer1_index] != 1:  # if presynaptic spike was not in the tolerance window, reduce weights of that synapse
            synapses[winner_index][layer1_index] = self.update_synapse(
                synapses[winner_index][layer1_index], self.STDP_weighting_curve(1))

    def get_winner_neuron(self, current_potentials, output_layer):
        winner_index = np.argmax(current_potentials)
        winner_neuron = output_layer[winner_index]
        return winner_index, winner_neuron

    def calculate_potentials_and_adapt_thresholds(self, current_potentials, neuron, neuron_index, spike_train, synapses,
                                                  time_step):
        if neuron.rest_until >= time_step:
            return

        # 更新膜电位
        neuron.potential += np.dot(synapses[neuron_index], spike_train[:, time_step])

        if neuron.potential > self.parameters.resting_potential:
            neuron.potential -= self.parameters.spike_drop_rate

            # 更新自适应阈值
            neuron.adaptive_spike_threshold = max(
                neuron.adaptive_spike_threshold - self.parameters.threshold_drop_rate,
                self.parameters.spike_threshold
            )

        # 存储当前膜电位
        current_potentials[neuron_index] = neuron.potential



