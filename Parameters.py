import pathlib
import plac

class Parameters:
    # Script Settings
    weights_path = "Pretrained/weights.csv"
    labels_path = "Pretrained/labels.csv"
    debugging_interval = 1
    checkpoint_interval = 1
    train_dataset_path = "MNIST/training/"
    test_dataset_path = "MNIST/testing/"
    training_images_amount = 1500
    test_images_amount = 100
    testing = True

    # Simulation Parameters
    image_train_time = 100  # Training time for every image
    past_window = -5
    epochs = 5

    # Input Parameters
    image_size = (28, 28)
    resting_potential = -70
    layer1_size = image_size[0] * image_size[1]  # Number of neurons in first layer
    layer2_size = 800  # Number of neurons in second layer

    # Neuron Parameters
    inhibitory_potential = -100
    spike_threshold = -55
    hyperpolarization_potential = -90
    spike_drop_rate = 0.8
    threshold_drop_rate = 0.4
    min_weight = 0.00001
    max_weight = 1.0

    # STDP Parameters
    sigma = 0.01  # STDP Learning Rate
    ltd_lr = 0.8  # Scaling Factor for Training Purposes - Decreasing Synpatic Weights
    ltp_lr = 0.3  # Scaling Factor for Training Purposes - Increasing Synpatic Weights
    mu = 0.9  # Exponential Factor

    # Spike Train Coding
    min_frequency = 1
    max_frequency = 50

    @plac.opt('weights_path', abbrev="weights_path",help="Path to weights file", type=pathlib.Path)
    @plac.opt('labels_path', abbrev="labels_path",help="Path to labels file", type=pathlib.Path)
    @plac.opt('debugging_interval', abbrev="debugging_interval", help="Script Setting", type=int)
    @plac.opt('checkpoint_interval', abbrev='checkpoint_interval', help="Script Setting", type=int)
    @plac.opt('train_dataset_path', abbrev='train_dataset_path', help="Script Setting", type=pathlib.Path)
    @plac.opt('test_dataset_path', abbrev='test_dataset_path', help="Script Setting", type=pathlib.Path)
    @plac.opt('training_images_amount', abbrev='training_images_amount', help="Script Setting", type=int)
    @plac.opt('test_images_amount', abbrev='test_images_amount', help="Script Setting", type=int)
    @plac.opt('testing', abbrev='testing', help="Script Setting", type=bool)
    @plac.opt('image_train_time', abbrev='image_train_time', help="Simulation Parameter", type=int)
    @plac.opt('past_window', abbrev='past_window', help="Simulation Parameter", type=int)
    @plac.opt('epochs', abbrev='epochs', help="Simulation Parameter", type=int)
    @plac.opt('image_size', abbrev='image_size', help="Input Parameter", type=tuple)
    @plac.opt('resting_potential', abbrev='resting_potential', help="Simulation Parameter", type=int)
    @plac.opt('layer2_size', abbrev='layer2_size', help="Simulation Parameter", type=int)
    @plac.opt('inhibitory_potential', abbrev='inhibitory_potential', help="Neuron Parameter", type=int)
    @plac.opt('spike_threshold', abbrev='spike_threshold', help="Neuron Parameter", type=int)
    @plac.opt('hyperpolarization_potential', abbrev='hyperpolarization_potential', help="Neuron Parameter", type=int)
    @plac.opt('spike_drop_rate', abbrev='spike_drop_rate', help="Neuron Parameter", type=float)
    @plac.opt('threshold_drop_rate', abbrev='threshold_drop_rate', help="Neuron Parameter", type=float)
    @plac.opt('min_weight', abbrev='min_weight', help="Neuron Parameter", type=float)
    @plac.opt('max_weight', abbrev='max_weight', help="Neuron Parameter", type=float)
    @plac.opt('STDP_offset', abbrev='STDP_offset', help="STDP Parameter", type=float)
    @plac.opt('sigma', abbrev='sigma', help="STDP Parameter", type=float)
    @plac.opt('ltd_lr', abbrev='ltd_lr', help="STDP Parameter", type=float)
    @plac.opt('ltp_lr', abbrev='ltp_lr', help="STDP Parameter", type=float)
    @plac.opt('mu', abbrev='mu', help="STDP Parameter", type=float)
    @plac.opt('min_frequency', abbrev='min_frequency', help="Spike Train Coding Parameter", type=float)
    @plac.opt('max_frequency', abbrev='max_frequency', help="Spike Train Coding Parameter", type=float)
    def __init__(self, weights_path=weights_path, labels_path=labels_path, debugging_interval=debugging_interval,
                 checkpoint_interval=checkpoint_interval, train_dataset_path=train_dataset_path,
                 test_dataset_path=test_dataset_path, training_images_amount=training_images_amount,
                 test_images_amount=test_images_amount, testing=testing, image_train_time=image_train_time,
                 past_window=past_window, epochs=epochs, image_size=image_size, resting_potential=resting_potential,
                 layer2_size=layer2_size, inhibitory_potential=inhibitory_potential, spike_threshold=spike_threshold,
                 hyperpolarization_potential=hyperpolarization_potential, spike_drop_rate=spike_drop_rate,
                 threshold_drop_rate=threshold_drop_rate, min_weight=min_weight, max_weight=max_weight, sigma=sigma,
                 ltd_lr=ltd_lr, ltp_lr=ltp_lr, mu=mu, min_frequency=min_frequency, max_frequency=max_frequency):
        self.weights_path = weights_path
        self.labels_path = labels_path
        self.debugging_interval = debugging_interval
        self.checkpoint_interval = checkpoint_interval
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.training_images_amount = training_images_amount
        self.test_images_amount = test_images_amount
        self.testing = testing
        self.image_train_time = image_train_time
        self.past_window = past_window
        self.epochs = epochs
        self.image_size = image_size
        self.resting_potential = resting_potential
        self.layer1_size = image_size[0] * image_size[1]  # Number of neurons in first layer
        self.layer2_size = layer2_size # if layer2_size <= training_images_amount else training_images_amount
        self.inhibitory_potential = inhibitory_potential
        self.spike_threshold = spike_threshold
        self.hyperpolarization_potential = hyperpolarization_potential
        self.spike_drop_rate = spike_drop_rate
        self.threshold_drop_rate = threshold_drop_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.sigma = sigma
        self.ltd_lr = ltd_lr
        self.ltp_lr = ltp_lr
        self.mu = mu
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency

