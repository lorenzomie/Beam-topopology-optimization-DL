import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from tensorflow import keras
from keras import layers
from train_vae_rec_model import VAE_REC, KL_WEIGHT, LEARNING_RATE, ARRAY_LENGTH 
class TNN_MULTILABEL(keras.Model):
    def __init__(self, output_shape, learning_rate, labels_weights, latent_dim=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.classifier = self.build_classifier(output_shape)
        self.learning_rate = learning_rate
        self.labels_weights = labels_weights
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.displacement_loss_tracker = keras.metrics.Mean(name="mass_loss")
        self.mass_loss_tracker = keras.metrics.Mean(name="stress_loss")
        self.frequency_loss_tracker = keras.metrics.Mean(name="frequency_loss")
        self.wave_time_loss_tracker = keras.metrics.Mean(name="wave_time_loss")

    def build_classifier(self, output_shape):
        # shape accepts a tuple of dimension
        classifier_inputs = keras.Input(shape = (self.latent_dim, ))
        x = layers.Dense(8, activation = "relu")(classifier_inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(16, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation = "relu")(x)
        classifier_outputs = layers.Dense(output_shape, activation = "softmax")(x)
        classifier = keras.Model(classifier_inputs, classifier_outputs, name = "Multilabel_Classifier")
        return classifier

    def call(self, inputs, training=None, mask=None):
        return self.classifier(inputs)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.displacement_loss_tracker,
            self.mass_loss_tracker,
            self.frequency_loss_tracker,
            self.wave_time_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            x, y = data
            x = tf.expand_dims(x, axis=0)
            y_pred = self.classifier(x)
            y = tf.expand_dims(y, axis = 0)
            displacement_loss = keras.losses.mean_squared_error(y_pred[:, 0], y[:, 0])*self.labels_weights[0]
            mass_loss = keras.losses.mean_squared_error(y_pred[:, 1], y[:, 1])*self.labels_weights[1]
            frequency_loss = keras.losses.mean_squared_error(y_pred[:, 2], y[:, 2])*self.labels_weights[2]
            wave_time_loss = keras.losses.mean_squared_error(y_pred[:, 3], y[:, 3])*self.labels_weights[3]
            total_loss = (displacement_loss + mass_loss + frequency_loss + wave_time_loss)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.displacement_loss_tracker.update_state(displacement_loss)
        self.mass_loss_tracker.update_state(mass_loss)
        self.frequency_loss_tracker.update_state(frequency_loss)
        self.wave_time_loss_tracker.update_state(wave_time_loss)
        return {
            "Total": self.total_loss_tracker.result(),
            "displacement": self.displacement_loss_tracker.result(),
            "mass": self.mass_loss_tracker.result(),
            "frequency": self.frequency_loss_tracker.result(),
            "wave_time": self.wave_time_loss_tracker.result(),
        }
    
    def test_step(self, data):
        x, y = data
        x = tf.expand_dims(x, axis=0)
        y = tf.expand_dims(y, axis = 0)
        y_pred = self(x, training=False)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

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

def split_dataset(vectors, labels, test_size, val_size, shuffle=True, seed=None):
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
        new_labels = labels[0:2]
        aux_labels = np.reshape(labels[2], (np.shape(labels[2])[1], np.shape(labels[2])[0]))
        for elem in aux_labels:
            new_labels.append(np.vstack(elem))
        new_labels.append(labels[3])
        labels = new_labels

    tf_vectors = tf.convert_to_tensor(vectors)
    labels = np.concatenate(labels, axis=1)
    tf_labels = tf.convert_to_tensor(labels)
    
    # Create a TensorFlow dataset for the data and labels
    dataset = tf.data.Dataset.from_tensor_slices((tf_vectors, tf_labels))

    # Shuffle the dataset if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=num_samples, seed=seed)

    # Split the dataset into training and test sets
    train_dataset = dataset.skip(num_test_samples+num_val_samples)
    test_val_dataset = dataset.take(num_test_samples+num_val_samples)
    test_dataset = test_val_dataset.skip(num_val_samples)
    validation_dataset = test_val_dataset.take(num_val_samples)

    return train_dataset, test_dataset, validation_dataset

def plot_dataset(data, labels):
    z_axis = ["displacement","mass","frequency","wave_time"]
    for idx, label in enumerate(labels):
        cmap = plt.get_cmap('viridis')
        label_colors = [cmap(label_value) for label_value in label]
        fig = plt.figure()
        plt.scatter(data[:,0], data[:,1], c=label_colors)
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(z_axis[idx])
        plt.show()

def save_weights_on_user_input(model, output_weight_path):
    print("Training Finished")
    ans = input("Are the results satisfactory?(y/n)")
    if ans.lower() == "y":
        model.save_weights(output_weight_path)

if __name__ == "__main__":
    ### PATH MANAGEMENT ###
    # Get the folder containing the script
    script_folder = os.path.dirname(os.path.abspath(__file__))

    # Define relative paths for training input folder
    dataset_path = os.path.join(script_folder, "..", "..", "data", "training", "multilabel_dataset.txt")
    weights_path = os.path.join(script_folder, "..", "..", "models", "VAE_REC", "vae.weights.h5")
    output_weights_path = os.path.join(script_folder, "..", "..", "models", "TNN", "tnn_multilabel.weights.h5")
    
    ### HYPERPARAMETER ###
    # ML stands for Multilabel
    NUM_EPOCHS_ML = 50
    BATCH_SIZE_ML = 20
    LEARNING_RATE_ML = 0.05
    LABELS_WEIGHTS = [10,1,10,1]
    
    ### MODEL TRAINING ###
    # Retrieve data from abaqus dataset txt file
    vectors, labels = get_data_from_dataset(dataset_path)
    
    # Loading the VAE used for the pretrain
    vae_rec = VAE_REC(ARRAY_LENGTH, KL_WEIGHT, LEARNING_RATE)
    vae_rec.load_weights(weights_path)
    _, _, z = vae_rec.encoder(vectors)
    
    # Select the frequencies of interest from the interval [1, 5]
    natural_frequencies = [1]
    labels[2] = select_frequencies(labels[2], natural_frequencies)
    
    # Scale to [0, 1] interval the labels, vectors are scaled previously
    labels_scaled, labels_min, labels_max = data_scaling(labels)
    labels_shape = len(labels)
    
    # Split train and test model
    TEST_SIZE = 0.1
    VALIDATION_SIZE = 0.1
    train_dataset, test_dataset, val_dataset= split_dataset(z, labels_scaled, TEST_SIZE, VALIDATION_SIZE)
    plot_dataset(z, labels_scaled)
    
    # Creating an Early Stopping for regularization (added to dropout in the NN)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Creating the Neural Network for classification
    model_multilabel = TNN_MULTILABEL(labels_shape, LEARNING_RATE_ML, LABELS_WEIGHTS)
    model_multilabel.compile(optimizer = tf.keras.optimizers.Adam(LEARNING_RATE_ML))
    # model_multilabel.fit(
    #     train_dataset, epochs=NUM_EPOCHS_ML, batch_size=BATCH_SIZE_ML, 
    #     validation_data=val_dataset, callbacks=[early_stopping]
    #     )
    model_multilabel.fit(
        train_dataset, epochs=NUM_EPOCHS_ML, batch_size=BATCH_SIZE_ML
        )
    save_weights_on_user_input(model_multilabel, output_weights_path)