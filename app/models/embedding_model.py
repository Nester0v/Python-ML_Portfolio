import tensorflow as tf
from app.models.dataset_loader import load_cifar10  # Import the load function


def create_embedding_model(input_shape):

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),  # Embedding layer
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == "__main__":
    # Load the CIFAR-10 dataset
    train_images, train_labels, test_images, test_labels = load_cifar10()

    # Create the embedding model
    input_shape = train_images.shape[1:]  # (32, 32, 3) for CIFAR-10
    model = create_embedding_model(input_shape)

    # Print the model summary
    model.summary()

    # Optionally, train the model
    model.fit(train_images, train_labels, epochs=16, batch_size=64, validation_data=(test_images, test_labels))
