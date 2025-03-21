
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Define a simple model
def create_model():
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Define a function to simulate distributed training with parameter servers
def distributed_training():
    # Number of workers and parameter servers
    num_workers = 2
    num_ps = 2

    # Create a cluster specification
    cluster = {
        "worker": ["worker0.example.com:2222", "worker1.example.com:2222"],
        "ps": ["ps0.example.com:2222", "ps1.example.com:2222"]
    }

    # Set up the environment for distributed training
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': cluster,
        'task': {'type': 'worker', 'index': 0}  # This can be 0 or 1 based on the worker
    })

    # Create the ParameterServerStrategy
    strategy = tf.distribute.experimental.ParameterServerStrategy()

    # Use the strategy for distributed training
    with strategy.scope():
        model = create_model()

    # Generate dummy data for training
    # (In a real-world scenario, replace this with actual data loading)
    train_data = np.random.random((1000, 784))
    train_labels = np.random.randint(10, size=(1000,))

    # Train the model
    model.fit(train_data, train_labels, epochs=5, batch_size=32)

# Start the distributed training
distributed_training()

