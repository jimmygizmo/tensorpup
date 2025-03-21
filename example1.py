
import tensorflow as tf

# Define the strategy
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

# Prepare dataset (this should be a dataset that is split across workers)
# For example, you could use tf.data API to partition your dataset
# Here we use MNIST for simplicity
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Prepare the dataset to be distributed across workers
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(60000).batch(256)

# Train the model
model.fit(train_dataset, epochs=5)

