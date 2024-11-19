import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Define paths
train_path = r"E:\Programs\Development\PyCharm\Python-ML_Portfolio\dataset\kaggle\brain_tumor_classification\Training"
test_path = r"E:\Programs\Development\PyCharm\Python-ML_Portfolio\dataset\kaggle\brain_tumor_classification\Testing"
embeddings_path = "embeddings/embeddings.npy"
model_save_path = "brain_tumor_model.keras"


# Helper function to load images and labels
def load_images_from_folder(folder_path, img_size=(128, 128)):
    images, labels = [], []
    class_names = sorted(os.listdir(folder_path))
    class_indices = {class_name: idx for idx, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_folder = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_folder):
            continue

        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            img = load_img(img_path, color_mode="grayscale", target_size=img_size)
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(class_indices[class_name])

    return np.array(images), np.array(labels)


# Load and preprocess data
def load_data():
    train_images, train_labels = load_images_from_folder(train_path)
    test_images, test_labels = load_images_from_folder(test_path)

    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=42, stratify=train_labels)

    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)


# Define the CNN model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.25),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


# Load data
(train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_data()

# Compute class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights_dict = dict(enumerate(class_weights))

# Instantiate and compile the model
model = create_model()

# Define callbacks
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True)

# Train the model
history = model.fit(train_data, train_labels,
                    epochs=50,
                    validation_data=(val_data, val_labels),
                    class_weight=class_weights_dict,
                    batch_size=16,
                    callbacks=[reduce_lr, early_stopping, model_checkpoint])


# Plot training history
def plot_history(history):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(history.history['accuracy'], label='Train Accuracy')
    axs[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axs[0].set_title('Model Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(loc='upper left')

    axs[1].plot(history.history['loss'], label='Train Loss')
    axs[1].plot(history.history['val_loss'], label='Validation Loss')
    axs[1].set_title('Model Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend(loc='upper right')

    plt.show()


plot_history(history)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_data, test_labels)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Predict on test set
y_pred = np.argmax(model.predict(test_data), axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(test_labels, y_pred, target_names=['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']))


# Confusion Matrix
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()


plot_confusion_matrix(test_labels, y_pred, labels=['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'])

# Save model
model.save(model_save_path)
print(f"Model saved to {model_save_path}")


def extract_embeddings(model, data, save_path):
    try:
        # Build the model if it's not built yet
        if not model.built:
            model.build(input_shape=(None, 128, 128, 1))  # Specify the input shape explicitly if not built

        # Perform a dummy prediction to initialize the model layers
        model.predict(data[:1])  # Passing just one sample to initialize the model

        # Identify the penultimate layer dynamically
        penultimate_layer = model.layers[-2]  # Assuming the second last layer is Dense for embeddings
        embedding_model = tf.keras.Model(inputs=model.input, outputs=penultimate_layer.output)
        embeddings = embedding_model.predict(data, batch_size=16)

        # Save embeddings to file
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, embeddings)
        print(f"Embeddings successfully saved to {save_path}")
    except Exception as e:
        print(f"Error extracting embeddings: {e}")
