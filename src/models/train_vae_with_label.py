import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers

class VAE_WITH_LABEL(keras.Model):
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
    def __init__(self, input_shape, kl_weight, svd_weight, learning_rate, latent_dim=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder(input_shape, latent_dim)
        self.decoder = self.build_decoder(input_shape)
        self.kl_weight = kl_weight
        self.svd_weight = svd_weight
        self.learning_rate = learning_rate
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.svd_loss_tracker = keras.metrics.Mean(name="svd_loss")

    def build_encoder(self, input_shape, latent_dim):
        encoder_inputs = keras.Input(shape = (input_shape,))
        x = layers.Dense(64, kernel_initializer='random_uniform', activation = 'relu')(encoder_inputs)
        # x = layers.Dense(32, kernel_initializer='random_uniform', activation="relu")(x)
        # x = layers.Dense(16, kernel_initializer='random_uniform', activation="relu")(x)
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
        decoder_inputs = keras.Input(shape = (self.latent_dim,))
        x = layers.Dense(16, kernel_initializer='random_uniform', activation = "relu")(decoder_inputs)
        # x = layers.Dense(32, kernel_initializer='random_uniform', activation="relu")(x)
        # x = layers.Dense(64, kernel_initializer='random_uniform', activation="relu")(x)
        decoder_outputs = layers.Dense(input_shape, activation = "relu")(x)
        decoder_outputs = CustomActivation(0.5, 0.55)(decoder_outputs)
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
            self.svd_loss_tracker,
        ]

    def train_step(self, batch_data):
        with tf.GradientTape() as tape:
            data, labels = batch_data
            labels = tf.cast(labels, dtype=tf.float32)
            # Check if a single label is input
            if labels.ndim == 1:
                labels = tf.expand_dims(labels, axis = 1)
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            # tf.print(data, reconstruction)
            reconstruction_loss = keras.losses.mean_squared_error(data, reconstruction)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))*self.kl_weight
            data_svd = tf.concat([z_mean, labels], axis=1)
            data_mean = tf.reduce_mean(data_svd, axis=0)
            data_reshaped = tf.reshape(data_svd - data_mean, [tf.shape(data)[0], -1]) 
            dd = tf.linalg.diag_part(tf.linalg.matmul(data_reshaped, data_reshaped, transpose_a=True))
            svd_loss = tf.reduce_sum(dd[1:] / dd[0])*self.svd_weight
            total_loss = reconstruction_loss + kl_loss + svd_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.svd_loss_tracker.update_state(svd_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "svd_loss": self.svd_loss_tracker.result(),
        }
class CustomActivation(tf.keras.layers.Layer):
    def __init__(self, threshold_1, threshold_2, **kwargs):
        super(CustomActivation, self).__init__(**kwargs)
        self.supports_masking = True
        self.threshold_1 = tf.constant(threshold_1, dtype = tf.float32)
        self.threshold_2 = tf.constant(threshold_2, dtype = tf.float32)

    def call(self, inputs):
        def element_activation(x):
            condition_1 = tf.math.less(x, self.threshold_1)
            condition_2 = tf.math.less(x, self.threshold_2)
            result = tf.where(condition_1, tf.constant(0.333333333, dtype = tf.float32), 
                            tf.where(condition_2, tf.constant(0.6666666666, dtype = tf.float32),
                                     tf.constant(1, dtype = tf.float32)))
            return result
        return tf.map_fn(element_activation, inputs)

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

    # The first line is the header, the counter starts after that
    for line in lines[1:]:
        row = line.split(';')
        vectors.append(np.array(row[:100], dtype=float))
        mass.append(np.array(row[100], dtype=float))
        stress.append(np.array(row[101], dtype=float))
        # The frequency string is "[0, f1, f1, ..., f5, f5]" alike 
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

def min_max_scaling(array, axis_custom=0):
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
        array_max = np.max(array, axis=axis_custom)
        array_min = np.min(array, axis=axis_custom)
        print("multidimensional array detected")
        scaled_array = np.zeros_like(array)

        for i in range(len(array_max)):
            if array_max[i] != array_min[i]:
                scaled_array[:, i] = (array[:, i] - array_min[i]) / (array_max[i] - array_min[i])
            else:
                scaled_array[:, i] = array[:, i]  # Avoid zero division
    else:
        raise ValueError("Not valid input. Only array and 2-D matrices accepted")

    return scaled_array, array_max, array_min

def split_dataset(vectors, labels, test_size, val_size, batch_size, shuffle=True, seed=None):
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
    num_val_samples = int(val_size * num_samples)
    print("The length of the training dataset is:", num_samples-num_test_samples-num_val_samples)
    print("The length of the test dataset is:", num_test_samples)
    print("The length of the validation dataset is:", num_val_samples)

    # Check if the Frequencies are an array or a single value (User choice)
    if all(len(elem) > 1 for elem in labels[2]):
        print("frequency are more than one")
        new_labels = labels[0:2]
        aux_labels = np.reshape(labels[2], (np.shape(labels[2])[1], np.shape(labels[2])[0]))
        for elem in aux_labels:
            new_labels.append(np.vstack(elem))
        new_labels.append(labels[3])
        labels = new_labels

    labels = np.concatenate(labels, axis=1)
    
    tf_dataset = tf.convert_to_tensor(vectors)
    
    # Create a TensorFlow dataset for the data and labels
    dataset = tf.data.Dataset.from_tensor_slices((tf_dataset, labels))

    # Shuffle the dataset if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=num_samples//10, seed=seed)

    # Split the dataset into training and test sets
    train_dataset = dataset.skip(num_test_samples+num_val_samples).batch(batch_size)
    test_val_dataset = dataset.take(num_test_samples+num_val_samples)
    test_dataset = test_val_dataset.skip(num_val_samples).batch(batch_size)
    validation_dataset = test_val_dataset.take(num_val_samples).batch(batch_size)
    
    return train_dataset, test_dataset, validation_dataset

def save_weights_on_user_input(model, output_weight_path):
    print("Training Finished")
    ans = input("Are the results satisfactory?(y/n)")
    if ans.lower() == "y":
        model.save_weights(output_weight_path)

def plot_latent_space_2D(vae_model, vectors, labels):
    z_axis = ["displacement","mass","frequency","wave_time"]
    tf_dataset = tf.convert_to_tensor(vectors)
    z_mean, _, z = vae_model.encoder(tf_dataset)
    cmap = plt.get_cmap('cividis')
    for idx, label in enumerate(labels):
        label_colors = [cmap(label_value) for label_value in label]
        fig = plt.figure()
        plt.scatter(z_mean[:, 0], z_mean[:, 1], c=label_colors)
        plt.colorbar()
        plt.xlabel('z_x')
        plt.ylabel('z_y')
        plt.title(z_axis[idx])
        plt.show()

if __name__ == "__main__":

    ### PATH MANAGEMENT ###
    file_Path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(file_Path, "..", "..", "data", "training", "multilabel_dataset_2.txt")
    output_weights_path = os.path.join(file_Path, "..", "..", "models", "VAE_WITH_LABEL", "vae_with_labels.weights.h5")

    ### HYPERPARAMETER TUNING ###
    NUM_EPOCHS = 100
    BATCH_SIZE = 50
    KL_WEIGHT = 0.001
    SVD_WEIGHT = 0.1
    LEARNING_RATE = 0.01

    ### MODEL TRAINING ###
    # Retrieve data from abaqus dataset txt file
    vectors, labels = get_data_from_dataset(dataset_path)
    vectors = vectors[:,0::4]
    print(vectors)

    # Select the frequencies of interest from the interval [1, 5]
    natural_frequencies = [1]
    labels[2] = select_frequencies(labels[2], natural_frequencies)

    # Scale to [0, 1] interval the labels, vectors are scaled previously
    labels_scaled, labels_min, labels_max = data_scaling(labels)
    input_shape = np.shape(vectors)[1]
    # print(labels_min, labels_max)

    # Create an unsupervised dataset
    TEST_SIZE = 0
    VALIDATION_SIZE = 0
    train_dataset, test_dataset, val_dataset = split_dataset(
        vectors, labels_scaled, TEST_SIZE, VALIDATION_SIZE, BATCH_SIZE, shuffle=True, seed=None
        )

    # print(train_dataset)
    # Create the VAE model
    vae_with_label = VAE_WITH_LABEL(input_shape, KL_WEIGHT, SVD_WEIGHT, LEARNING_RATE)

    # Display summaries of the encoder and decoder
    print("Encoder Summary:")
    vae_with_label.encoder.summary()
    print("\nDecoder Summary:")
    vae_with_label.decoder.summary()

    # Compile and train the VAE model
    vae_with_label.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
    vae_with_label.fit(train_dataset, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

    for i in range(10):
        print( "vector", vectors[i,:], vae_with_label(vectors[i:i+1,:]))
    # plot_latent_space_2D(vae_with_label, vectors, labels_scaled)

    save_weights_on_user_input(vae_with_label, output_weights_path)
