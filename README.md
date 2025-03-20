# Multilayer Perceptron for Human Activity Recognition (HAR)

This project implements a Multilayer Perceptron (MLP) to classify human activities based on the UCI HAR dataset. The dataset contains sensor data collected from smartphones, and the goal is to predict one of six activities performed by participants.

## Features
- **Leave-One-Participant-Out Cross-Validation**: The model is trained on data from all participants except one, which is used for testing.
- **Custom Neural Network Architecture**:
  - Input layer with 561 neurons (features).
  - Hidden layer with 300 neurons.
  - Output layer with 6 neurons (activities).
- **Error Backpropagation**: Implements forward and backward passes to train the network.
- **Confusion Matrix**: Evaluates the model's performance by computing a confusion matrix.

## Usage
1. Ensure the `uci_har.csv` dataset is in the project directory.
2. Run the `main.py` script to train and evaluate the model.
3. The confusion matrix will be printed at the end of the execution.

## Requirements
- Python 3.11/3.12
- NumPy
- Pandas

## Dataset
The UCI HAR dataset is used for this project. It contains:
- Sensor readings as features.
- Activity labels (1-6).
- Participant IDs for cross-validation.

## Disclaimer
This project is for educational purposes only and is not intended for production use.


