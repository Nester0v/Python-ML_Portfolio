import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

def load_cifar10():
    # Load CIFAR-10 dataset from TensorFlow
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    # Normalize the images to the range [0, 1]
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    return train_images, train_labels, test_images, test_labels

if __name__ == "__main__":
    # For testing the function
    train_images, train_labels, test_images, test_labels = load_cifar10()
    print(f"Train Images Shape: {train_images.shape}")
    print(f"Train Labels Shape: {train_labels.shape}")
    print(f"Test Images Shape: {test_images.shape}")
    print(f"Test Labels Shape: {test_labels.shape}")
