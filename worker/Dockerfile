
# Use official TensorFlow image as base
FROM tensorflow/tensorflow:2.12.0-gpu

# Set the working directory inside the container
WORKDIR /app

# Install any additional dependencies (optional)
RUN pip install --upgrade pip
RUN pip install tensorflow

# Set environment variables (This is where you can configure TF_CONFIG)
# For example, setting worker task and role using the TF_CONFIG environment variable
# The TF_CONFIG variable must be set appropriately depending on the worker's role
# Here, we use a placeholder that will be replaced at runtime by the cluster manager

ENV TF_CONFIG='{
  "cluster": {
    "worker": ["worker1.example.com:1234", "worker2.example.com:1234"]
  },
  "task": {"type": "worker", "index": 0}
}'

# Copy the Python training script into the container
COPY train.py /app/train.py

# Expose the necessary ports for communication (for example, if you use ports for cluster communication)
EXPOSE 1234  # Replace with actual port numbers used by your setup

# Command to run the training script
CMD ["python", "train.py"]

