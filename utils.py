import tensorflow as tf

def prepare_data(snn):
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
    X_train, Y_train = X_train[:snn.parameters.training_images_amount], Y_train[
                                                                         :snn.parameters.training_images_amount]
    X_test, Y_test = X_test[:snn.parameters.test_images_amount], Y_test[:snn.parameters.test_images_amount]

    return X_train, Y_train, X_test, Y_test