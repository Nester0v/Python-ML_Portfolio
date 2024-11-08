# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
#
# class BrainTumorDataset:
#     def __init__(self, train_dir, test_dir, target_size=(150, 150), batch_size=32):
#         self.train_dir = train_dir
#         self.test_dir = test_dir
#         self.target_size = target_size
#         self.batch_size = batch_size
#         self.train_generator = None
#         self.test_generator = None
#
#     def load_data(self):
#         # Data Augmentation for Training
#         train_datagen = ImageDataGenerator(
#             rescale=1.0/255,
#             rotation_range=20,
#             brightness_range=[0.8, 1.2],
#             zoom_range=0.2,
#             horizontal_flip=True,
#             vertical_flip=True
#         )
#         # Only Rescaling for Testing
#         test_datagen = ImageDataGenerator(rescale=1.0/255)
#
#         # Load Images from Directories
#         self.train_generator = train_datagen.flow_from_directory(
#             self.train_dir,
#             target_size=self.target_size,
#             batch_size=self.batch_size,
#             class_mode='binary'
#         )
#         self.test_generator = test_datagen.flow_from_directory(
#             self.test_dir,
#             target_size=self.target_size,
#             batch_size=self.batch_size,
#             class_mode='binary'
#         )
#         return self.train_generator, self.test_generator
