Brain Tumor Classification Using Convolutional Neural Networks

This repository contains a TensorFlow-based implementation of a Convolutional Neural Network (CNN) for classifying brain tumors into four categories: Glioma, Meningioma, No Tumor, and Pituitary Tumor.

Key Features

Data Handling: Load, preprocess, and augment grayscale brain MRI images from structured folders.
Model Architecture: A deep CNN with batch normalization and dropout layers for robust feature extraction and overfitting prevention.

Training Optimization:
-Class Weights: Address imbalanced data using computed class weights.
-Callbacks: Incorporates learning rate reduction, early stopping, and model checkpointing.
-Performance Visualization:Training history plots for accuracy and loss.

Confusion matrix and classification reports for detailed evaluation.
Embeddings Extraction: Generate feature embeddings from the penultimate layer for further downstream tasks.

Setup and Usage

-Dataset: Organize training and testing data in separate folders with subfolders for each class.
-Paths: Update paths to your dataset and output files in the script.
-Run: Execute the script to train, evaluate, and save the model.

Results

-Visualization: Gain insights into model performance through accuracy/loss curves and confusion matrices.
-Metrics: Evaluate the model's precision, recall, and F1-score for each class.

Applications
This project can be extended for:

-Feature-based clustering or analysis using extracted embeddings.
-Deployment of the trained model for real-time classification in medical imaging systems.

Also for Back-End used Flask for one page web app

Thanks for Sartaj  from Kaggle for dataset https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri
