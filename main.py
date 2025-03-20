# Implementation of multilayer perceptron
import numpy as np
import pandas as pd

# Load data
data = pd.read_csv('uci_har.csv')
activities = data.iloc[:, 0].values
participants = data.iloc[:, 1].values
features = data.iloc[:, 2:].values

# Network architecture as specified in part (a)
num_input = 561     # Input neurons
num_hidden = 300  # Hidden neurons
num_output = 6      # Output neurons (activities)
eta = 0.001        # Learning rate

## Training and evaluation with leave-one-participant-out
total_conf = np.zeros((num_output, num_output))

for test_participant in range(1, 31):
    # Split data according to participant
    train_idx = participants != test_participant
    test_idx = participants == test_participant
# Prepare training data
    Training = features[train_idx].T
    train_activities = activities[train_idx]
# Prepare test data
    Test = features[test_idx].T
    test_activities = activities[test_idx]
    
# Create targets (-0.9/0.9 as in your example)

    TrainingTargets = -0.9 * np.ones((num_output, np.sum(train_idx)))
    for k in range(len(train_activities)):
        TrainingTargets[train_activities[k]-1, k] = 0.9

# Initialize weights (same as your example)

    W_hat = 0.02 * (np.random.rand(num_hidden, num_input + 1) - 0.5)
    W = 0.02 * (np.random.rand(num_output, num_hidden + 1) - 0.5)
    
    E = float('inf')
    epoch = 0
# Training loop
    while epoch < 100:
        E = 0
        for l in range(Training.shape[1]):
            # Forward pass
            v_hat = W_hat @ np.concatenate(([1], Training[:, l]))
            z = np.tanh(v_hat)
            v = W @ np.concatenate(([1], z))
            o = np.tanh(v)
            
        # Error and backpropagation
            E += np.sum((o - TrainingTargets[:, l])**2)
            delta = (o - TrainingTargets[:, l]) * (1 - o**2)
            intermediatestep = []
            for i in range(W.shape[1]):  
                value = 0
                for j in range(W.shape[0]):  
                    value += W[j, i] * delta[j] 
                intermediatestep.append(value)
            intermediatestep = np.array(intermediatestep)  
            delta_hat = intermediatestep[1:] * (1 - z**2)
            

            # Weight updates
            W = W - eta * np.outer(delta, np.concatenate(([1], z)))
            W_hat = W_hat - eta * np.outer(delta_hat, np.concatenate(([1], Training[:, l])))
        epoch += 1
        print(epoch)
        E = E / Training.shape[1]
    
    # Testing and confusion matrix computation
    conf = np.zeros((num_output, num_output))
    for k in range(Test.shape[1]):
        z = np.tanh(W_hat @ np.concatenate(([1], Test[:, k])))
        out = np.tanh(W @ np.concatenate(([1], z)))
        pred = np.argmax(out)
        true = test_activities[k] - 1
        conf[true, pred] += 1
    total_conf += conf

print("Confusion Matrix:")
print(total_conf)
