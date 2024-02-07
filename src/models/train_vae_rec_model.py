import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

ARRAY_LENGTH = 100 # This number is not changeable without changing the abaqus script
MATERIALS_NUMBER = 3 # This number is not changeable without changing the abaqus script
DIM = 1000000

# Hyperparameter tuning
NUM_EPOCHS = 100
BATCH_SIZE = 50
KL_WEIGHT = 0
LEARNING_RATE = 0.005

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
    def __init__(self, input_shape, kl_weight, learning_rate, latent_dim=2):
        super(VAE_REC, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder(input_shape, latent_dim)
        self.decoder = self.build_decoder(input_shape)
        self.kl_weight = kl_weight
        self.learning_rate = learning_rate
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def build_encoder(self, input_shape, latent_dim):
        encoder_inputs = keras.Input(shape = (input_shape,))
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
        decoder_inputs = keras.Input(shape = (self.latent_dim,))
        x = layers.Dense(16, activation = "relu")(decoder_inputs)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dense(64, activation = "relu")(x)
        decoder_outputs = layers.Dense(input_shape, activation = "lu")(x)
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

def generate_constrained_vector(length, max_value, max_unique_numbers=3):
    """
    Generate a constrained vector.

    Parameters:
    - length (int): Length of the vector.
    - max_value (int): Maximum value for the elements in the vector.
    - max_unique_numbers (int): Maximum number of unique numbers in the vector.

    Returns:
    np.ndarray: Normalized vector.
    """
    if max_unique_numbers > max_value:
        raise ValueError("The maximum number of unique numbers cannot exceed the maximum value.")

    unique_numbers = np.random.choice(max_value, size=min(max_unique_numbers, max_value), replace=False)
    random_vector = np.random.choice(unique_numbers, size=length, replace=True)
    normalized_vector = random_vector / max_value
    return normalized_vector

def create_dataset(dimension):
    """
    Create a dataset with random vectors.

    Parameters:
    - dimension (int): Number of vectors in the dataset.

    Returns:
    List[np.ndarray]: List of random vectors.
    """
    dataset = []
    for _ in range(dimension):
        array = generate_constrained_vector(ARRAY_LENGTH, MATERIALS_NUMBER)
        dataset.append(array)
    return dataset

def save_model(path):
    """Save the VAE model weights to the specified file path."""
    if os.path.exists(path):
        ans = input("The model weights.vae already exist, want to overwrite it (y/n): ")
        if ans.lower() =='y':
            os.remove(path)

    vae.save_weights(path)
    print("weights saved.")

if __name__ == "__main__":
    
    file_Path = os.path.abspath(__file__)
    output_weight_path = r"../../../models/VAE_REC"
    # Generate the dataset
    dataset = create_dataset(dimension=DIM)
    print(dataset)

    # Convert the dataset to a TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    train_dataset = dataset.shuffle(DIM).batch(BATCH_SIZE)
    print(train_dataset)
    # Create the VAE model
    vae = VAE_REC(ARRAY_LENGTH, KL_WEIGHT, LEARNING_RATE)

    # Display summaries of the encoder and decoder
    print("Encoder Summary:")
    vae.encoder.summary()
    print("\nDecoder Summary:")
    vae.decoder.summary()

    # Compile and train the VAE model
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE))
    vae.fit(train_dataset, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

    output_path = output_weight_path + '\\vae_rec.weights.h5'
    weights_path = os.path.abspath(os.path.join(file_Path, output_path))
    print(weights_path)
    save_model(weights_path)