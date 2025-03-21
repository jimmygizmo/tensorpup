
import tensorflow as tf
import os
import json

# Define the distribution strategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Build the model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

# Open and distribute the model under the strategy
with strategy.scope():
    model = build_model()

# Load dataset (use any dataset that suits your needs)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Prepare dataset to be distributed across workers
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(60000).batch(256)

# Train the model
model.fit(train_dataset, epochs=5)

