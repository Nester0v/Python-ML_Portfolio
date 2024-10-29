import pickle
import numpy as np
import os

# Define the CIFAR-10 directory
cifar10_dir = 'E:\\Programs\\Development\\PyCharm\\Python-ML_Portfolio\\dataset\\cifar-10-batches-py'

def load_batch(filepath):
    with open(filepath, 'rb') as file:
        batch = pickle.load(file, encoding='bytes')
        images = batch[b'data']
        labels = batch[b'labels']

        # CIFAR-10 images are stored in a flat format (32*32*3)
        images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return images, labels

# Load all batches
images_list = []
labels_list = []

for i in range(1, 6):
    batch_file = os.path.join(cifar10_dir, f'data_batch_{i}')
    images, labels = load_batch(batch_file)
    images_list.append(images)
    labels_list.extend(labels)

# Combine all batches into a single dataset
all_images = np.concatenate(images_list)
all_labels = np.array(labels_list)

print(f"Loaded CIFAR-10 dataset: {all_images.shape[0]} images with shape {all_images.shape[1:]}")
