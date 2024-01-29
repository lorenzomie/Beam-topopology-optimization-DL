import os
import numpy as np
from itertools import chain
import tensorflow as tf
from tensorflow import keras
from keras import layers

class VAE_REC(keras.Model):
    """Variational Autoencoder (VAE) implementation.

    This class defines a Variational Autoencoder model using TensorFlow and Keras.
    The VAE consists of an encoder, a decoder, and a sampling layer for the latent space.

    Args:
        input_shape (tuple): The shape of the input data.
        latent_dim (int): The dimension of the latent space (default is 2).

    Attributes:
        latent_dim (int): The dimension of the latent space.
        encoder (tf.keras.Model): The encoder model.
        decoder (tf.keras.Model): The decoder model.

    Methods:
        build_encoder(input_shape, latent_dim): Build the encoder model.
        sampling(args): Sample from the latent space.
        build_decoder(input_shape): Build the decoder model.
        train_step
        compile_and_train(data, num_epochs, batch_size): Compile and train the VAE.
    """
    def __init__(self, input_shape, kl_weight, learning_rate, labels, latent_dim=2):
        super(VAE_REC, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder(input_shape, latent_dim)
        self.decoder = self.build_decoder(input_shape)
        self.kl_weight = kl_weight
        self.learning_rate = learning_rate
        self.labels = labels
        self.mass_label = labels[0]
        self.stress_label = labels[1]
        self.frequencies_label = labels[2]
        self.wave_label = labels[3]
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def build_encoder(self, input_shape, latent_dim):
        encoder_inputs = keras.Input(shape = input_shape)
        x = layers.Dense(64, activation = 'relu')(encoder_inputs)
        x = layers.Dense(32, activation = 'relu')(x)
        x = layers.Dense(16, activation="relu")(x)
        z_mean = layers.Dense(latent_dim, name = "z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name = "z_log_var")(x)
        z = self.sampling([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="ENC")
        return encoder

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim)) # mean is 0 and sigma = 1
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def build_decoder(self, input_shape):
        # shape accepts a tuple of dimension
        decoder_inputs = keras.Input(shape = (self.latent_dim+len(self.labels),))
        x = layers.Dense(16, activation = "relu")(decoder_inputs)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dense(64, activation = "relu")(x)
        decoder_outputs = layers.Dense(input_shape, activation = "relu")(x)
        decoder = keras.Model(decoder_inputs, decoder_outputs, name = "DEC")
        return decoder

    def call(self, inputs, training=None, mask=None):
        _, _, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = keras.losses.mse(data, reconstruction)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

def get_data_from_dataset(dataset_path):
    """
    Reads data from an abaqus dataset txt file and extracts vectors, mass, stress, 
    frequencies, and wave time.

    Args:
        dataset_path (str): The path to the dataset file.

    Returns:
        vectors (ndarray): An array of vectors extracted from the dataset.
        labels (list of ndarrays): A list containing arrays for mass, stress, frequencies, and wave time.
    
    You have to run data_abaqus/make_multilabel_dataset.py before
    """
    vectors = []
    mass = []
    stress = []
    frequencies = []
    wave_time = []
    
    with open(dataset_path, 'r') as file:
        lines = file.readlines()

    for line in lines[1:]:
        row = line.split(';')
        vectors.append(np.array(row[:100], dtype=float))
        mass.append(np.array(row[100], dtype=float))
        stress.append(np.array(row[101], dtype=float))
        # The string is "[0, f1, f1, ..., f5, f5]" alike 
        frequencies_values = row[102].strip('[]').split(', ')[1::2] 
        frequencies.append(np.array(frequencies_values, dtype=float))
        wave_time.append(np.array(row[103][:-2], dtype=float )) # delete the endline \n
    
    vectors = np.vstack(vectors)
    mass = np.vstack(mass)
    stress = np.vstack(stress)
    frequencies = np.vstack(frequencies)
    wave_time = np.vstack(wave_time)
    
    return vectors, [mass, stress, frequencies, wave_time]

def select_frequencies(frequencies, request_array, all = False):
    """
    Selects frequencies from a list based on a request array containing the natural frequency of interest. 

    Args:
        frequencies (list): A list of natural frequencies to select from.
        request_array (list or set): An array containing the indices of the frequencies to select.
        all (bool, optional): If True, selects all frequencies. If False, selects 
            only the frequencies specified in the request array. Default is False.

    Returns:
        requested_frequencies(np.array): An array of selected frequencies based on the request array.

    Example:
        >> frequencies = [2, 14, 50, 90, 120]
        >> request_array = [2, 4]
        >> select_frequencies(frequencies, request_array)
        [14, 90]
    """
    # substituting None with the numbers outside the interval
    request_array = [None if num > 5 or num < 1 else int(num) for num in request_array]
    request_array = list(filter(None, request_array))
    
    if not all:
        requested_frequencies = frequencies[:, request_array] 
    else:
        requested_frequencies = frequencies
    
    return requested_frequencies

def min_max_scaling(array, axis=0):
    """
    Scales the input array using min-max scaling along the specified axis.

    Args:
        array (ndarray): The input array to be scaled.
        axis (int, optional): The axis along which to scale the array (default is column-wise).

    Returns:
        scaled_array (ndarray): The scaled array.
        array_max (ndarray): The maximum values along the specified axis.
        array_min (ndarray): The minimum values along the specified axis.
    """
    if array.shape[0] == 1 or array.shape[1] == 1:
        array_max = np.max(array)
        array_min = np.min(array)
        
        if array_max != array_min:
            scaled_array = (array - array_min) / (array_max - array_min)
        else:
            scaled_array = array # Avoid a zeros array

    elif array.shape[0] > 1 and array.shape[1] > 1:
        # manage a matrix case doing scaling along a column
        array_max = np.max(array, axis=axis)
        array_min = np.min(array, axis=axis)

        scaled_array = np.zeros_like(array)

        for i in range(len(array_max)):
            if array_max[i] != array_min[i]:
                scaled_array[:, i] = (array[:, i] - array_min[i]) / (array_max[i] - array_min[i])
            else:
                scaled_array[:, i] = array[:, i]  # Avoid zero division
    else:
        raise ValueError("Not valid input. Only array and 2-D matrices accepted")

    return scaled_array, array_max, array_min

def data_scaling(dataset):
    """
    Scales the data in the dataset using min-max scaling.

    Args:
        dataset (list): A list containing the data to be scaled.

    Returns:
        normalized_dataset (list): A list of scaled data.
        array_min (list): A list containing the minimum values for each feature.
        array_max (list): A list containing the maximum values for each feature.
    """
    normalized_dataset = []
    array_min = []
    array_max = []

    for data in dataset:
        normalized_data, data_min, data_max = min_max_scaling(data)
        normalized_dataset.append(normalized_data)
        array_min.append(data_min)
        array_max.append(data_max)
    
    return normalized_dataset, array_min, array_max

def split_train_test_dataset(vectors, labels, test_size, shuffle=True, seed=None):
    """
    Splits the dataset into training and test sets using TensorFlow.

    Args:
        vectors (numpy.ndarray): Numpy array of input vectors.
        labels_scaled (list): List of numpy arrays of normalized labels.
        test_size (float): Size of the test set (default 0.2).
        shuffle (bool): Whether to shuffle the dataset (default True).
        seed (int): Seed for shuffling the data (default None).

    Returns:
        train_dataset (tf.data.Dataset): Training dataset.
        test_dataset (tf.data.Dataset): Test dataset.
    """
    # Calculate the size of the test set
    num_samples = len(vectors)
    num_test_samples = int(test_size * num_samples)
    print("The length of the training dataset is:", num_samples-num_test_samples)
    print("The length of the test dataset is:", num_test_samples)
    
    # Check if the Frequencies are an array or a single value (User choice)
    if all(len(elem) > 1 for elem in labels[2]):
        new_labels = labels[0:2]
        aux_labels = np.reshape(labels[2], (np.shape(labels[2])[1], np.shape(labels[2])[0]))
        for elem in aux_labels:
            new_labels.append(np.vstack(elem))
        new_labels.append(labels[3])
        labels = new_labels

    tf_vectors = tf.convert_to_tensor(vectors)
    tf_labels = tf.convert_to_tensor(labels)
    
    # Create a TensorFlow dataset for the data and labels
    dataset = tf.data.Dataset.from_tensor_slices((tf_vectors, *tf_labels))

    # Shuffle the dataset if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=num_samples, seed=seed)

    # Split the dataset into training and test sets
    train_dataset = dataset.skip(num_test_samples)
    test_dataset = dataset.take(num_test_samples)

    return train_dataset, test_dataset

if __name__ == "__main__":
    ### PATH MANAGEMENT ###
    
    # Get the folder containing the script
    script_folder = os.path.dirname(os.path.abspath(__file__))

    # Define relative paths for training input folder
    dataset_path = os.path.join(script_folder, "..", "..", "data", "training", "multilabel_dataset.txt")
    
    ### HYPERPARAMETER ###
    
    NUM_EPOCHS = 100
    BATCH_SIZE = 20
    KL_WEIGHT = 0.05
    LEARNING_RATE = 0.005
    
    ### MODEL TRAINING ###
    # Retrieve data from abaqus dataset txt file
    vectors, labels = get_data_from_dataset(dataset_path)
    
    # Select the frequencies of interest from the interval [1, 5]
    natural_frequencies = [1, 2]
    labels[2] = select_frequencies(labels[2], natural_frequencies)
    
    # Scale to [0, 1] interval the labels, vectors are scaled previously
    labels_scaled, labels_min, labels_max = data_scaling(labels)
    
    print(labels_scaled)
    # Split train and test model
    TEST_SIZE = 0.2
    split_train_test_dataset(vectors, labels_scaled, TEST_SIZE)
    