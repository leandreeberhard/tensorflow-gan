import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# import mnist data
data = pd.read_csv('train.csv')
labels = data['label']
del data['label']
features = data.values

# transform the features to be between -1 and 1
scaler = MinMaxScaler(feature_range=(-1, 1))
features = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)

# training parameters
batch_size = 32
learning_rate = 0.0001
epochs = 50
training_steps = 75000
display_step = 100

# load train data into a tf pipeline
train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_data = train_data.repeat().shuffle(1000).batch(batch_size).prefetch(1)

# discriminator network specification
discriminator_f = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1024, input_shape=(784,), activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(265, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

# summary of model and trainable parameters
discriminator_f.summary()

# generator network specification
generator_f = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, input_shape=(100, ), activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(784, activation='tanh')
])

# generator summary
generator_f.summary()

# test if outputs can be calculated through the networks
discriminator_f(X_train)
generator_f(np.random.normal(0, 1, 100).reshape((1, 100)))

# optimizers, one for each network
optimizer_discriminator = tf.optimizers.Adam(learning_rate)
optimizer_generator = tf.optimizers.Adam(learning_rate)

# loss function
loss_function = tf.losses.BinaryCrossentropy()

# backward pass for both networks at once
def backward(x, y):
    # produce training data for the discriminator
    real_samples = x
    real_labels = y
    real_labels = np.ones(x.shape[0])

    random_noise = np.random.normal(0, 1, 100 * x.shape[0]).reshape(x.shape[0], 100)
    fake_samples = tf.cast(generator_f(random_noise), 'float64')
    fake_labels = np.zeros(x.shape[0])

    full_samples = tf.concat((real_samples, fake_samples), axis=0)
    full_labels = tf.concat((real_labels, fake_labels), axis=0)

    # optimize discriminator
    with tf.GradientTape() as g_discriminator:
        predicted_labels = discriminator_f(full_samples)
        loss_discriminator = loss_function(full_labels, predicted_labels)

    gradients_discriminator = g_discriminator.gradient(loss_discriminator, discriminator_f.variables)
    optimizer_discriminator.apply_gradients(zip(gradients_discriminator, discriminator_f.variables))

    # produce training data for the generator
    random_noise = np.random.normal(0, 1, 100 * x.shape[0]).reshape(x.shape[0], 100)

    # optimize generator
    with tf.GradientTape() as g_generator:
        fake_samples = generator_f(random_noise)
        predicted_labels_from_fake = discriminator_f(fake_samples)
        loss_generator = loss_function(real_labels.reshape((x.shape[0], 1)), predicted_labels_from_fake)

    gradients_generator = g_generator.gradient(loss_generator, generator_f.variables)
    optimizer_generator.apply_gradients(zip(gradients_generator, generator_f.variables))

    return loss_discriminator, loss_generator


# training loop
for step, (X_batch, y_batch) in enumerate(train_data.take(training_steps), 1):
    loss_discriminator, loss_generator = backward(X_batch, y_batch)

    if step % display_step ==  0:
        print(f'Epoch: {step}')
        print(f'Discriminator Loss: {loss_discriminator}')
        print(f'Generator Loss: {loss_generator}')



# make new samples from the generator
random_noise = np.random.normal(0, 1, 100 * 16).reshape(16, 100)
# generated figures
generated_samples = generator_f(random_noise)

# transform the data back to the original scale
generated_samples = scaler.inverse_transform(generated_samples)

# plot these samples
# plot samples
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(generated_samples[i].reshape(28, 28), cmap='gray_r', interpolation='nearest')
    plt.xticks([])
    plt.yticks([])


# plot some real digits for comparison
n_rows = X_train.shape[0]
random_indices = np.random.choice(n_rows, size=16, replace=False)

real_samples = X_train[random_indices, :]

real_samples = scaler.inverse_transform(real_samples)

for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(real_samples[i].reshape(28, 28), cmap='gray_r', interpolation='nearest')
    plt.xticks([])
    plt.yticks([])



