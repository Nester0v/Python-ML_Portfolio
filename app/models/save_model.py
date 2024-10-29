import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

# Load the CIFAR-10 dataset
(train_images, _), (_, _) = cifar10.load_data()

# Normalize images
train_images = train_images.astype('float32') / 255.0

# Define and compile your model (example using a simple CNN)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model (example, adjust epochs and batch_size as needed)
model.fit(train_images, epochs=10, batch_size=64)

# Save the trained model
model.save('cifar10_embedding_model.h5')
print("Model saved to 'cifar10_embedding_model.h5'")

# Generate and save embeddings for the training images
embeddings = model.predict(train_images)
np.save('embeddings/embeddings.npy', embeddings)  # Adjust the path as necessary
print("Embeddings saved to 'embeddings/embeddings.npy'")
