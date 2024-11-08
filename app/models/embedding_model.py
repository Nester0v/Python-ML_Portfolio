import numpy as np
import tensorflow as tf
import os


def load_data(train_dir, test_dir):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    train_data = datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )
    test_data = datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )
    return train_data, test_data


def create_embedding_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4, activation='softmax'),  # Output layer for 4 classes
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',  # Use categorical_crossentropy for multi-class classification
        metrics=['accuracy']
    )
    return model


def augment_data(train_images):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.1),
    ])
    return data_augmentation(train_images)


if __name__ == "__main__":
    train_dir = r'E:\Programs\Development\PyCharm\Python-ML_Portfolio\dataset\kaggle\brain_tumor_classification\Training'
    test_dir = r'E:\Programs\Development\PyCharm\Python-ML_Portfolio\dataset\kaggle\brain_tumor_classification\Testing'

    # Load the data
    train_data, test_data = load_data(train_dir, test_dir)

    # Define the input shape
    input_shape = (150, 150, 3)

    model_path = 'brain_tumor_model.keras'

    # Check if the model exists
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print("Loaded existing model.")
    else:
        model = create_embedding_model(input_shape)
        print("Created a new model.")

    # Callbacks
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train the model
    model.fit(train_data, epochs=25, validation_data=test_data, callbacks=[reduce_lr, early_stopping])

    # Save the trained model
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Generate and save embeddings for the training images
    embeddings = model.predict(train_data)
    np.save('embeddings/embeddings.npy', embeddings)
    print("Embeddings saved to 'embeddings/embeddings.npy'")
