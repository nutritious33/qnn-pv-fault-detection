import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from qiskit.circuit.library import EfficientSU2, PauliFeatureMap
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import Sampler
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.circuit.library import QNNCircuit
import time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# load data
original_features = np.loadtxt('fault.txt')
labels  = np.loadtxt('faultlabel.txt')

features = original_features[:, [0,5,6]]

scaler_x = MinMaxScaler()
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

f_train = scaler_x.fit_transform(X_train)
f_test = scaler_x.transform(X_test)

num_q = f_train.shape[1]
#fm = ZZFeatureMap(num_qubits, reps=2, entanglement='full')
fm = PauliFeatureMap(num_q, reps=2, paulis=['Z','H'])
ansatz = EfficientSU2(num_q, reps=4, entanglement='full', su2_gates=['ry','rz','h'])

# create quantum circuit
qc = QNNCircuit(
    num_qubits=num_q,
    feature_map=fm,
    ansatz=ansatz
)

# visualize quantum circuit
qc.decompose().draw('mpl', style='clifford')

def interpret(bitstring):
    return int(bitstring) % 2

sampler = Sampler()

# Qiskit quantum neural network
qnn = SamplerQNN(
    circuit = qc,
    input_params = fm.parameters,
    weight_params = ansatz.parameters,
    output_shape = 2,
    interpret = interpret,
    sampler=sampler
)

# Create a PyTorch model using TorchConnector
model = TorchConnector(qnn)
# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Convert training data to PyTorch tensors
X_train_tensor = torch.tensor(f_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)  
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
batch_size = 7
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
epochs = 15
for epoch in range(epochs):
    start_time = time.time()  # Record the start time
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = F.mse_loss(output, y_batch)
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    end_time = time.time()  # Record the end time
    epoch_duration = (end_time - start_time) / 60  # time in minutes
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader)}, Time: {epoch_duration:.2f} minutes')

# Convert test data to PyTorch tensors
X_test_tensor = torch.tensor(f_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Evaluate the model
model.eval()
with torch.no_grad():
    output = model(X_test_tensor)
    test_loss = F.mse_loss(output, y_test_tensor)
    predictions = torch.argmax(output, dim=1)
    actual = torch.argmax(y_test_tensor, dim=1)
    accuracy = (predictions == actual).float().mean()

print(f'Test set loss: {test_loss.item()}')
print(f'Test set accuracy: {accuracy.item() * 100:.2f}%')

# Confusion Matrix and Heatmap
cm = confusion_matrix(actual, predictions)

l = ['Fault', 'No Fault']
column = [f'{label}' for label in l]
indices = [f'{label}' for label in l]
table = pd.DataFrame(cm, columns=column, index=indices)

print("Confusion Matrix")
print(cm)

# Normalize by truth labels
table_normalized = table.div(table.sum(axis=1), axis=0)

# Plotting the heatmap
color = None
sns.heatmap(table_normalized, annot=True, fmt='.2%', cmap=color)
plt.title("Quantum Multi-class Fault Detection", fontsize=15)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.rcParams.update({'font.size': 17})
plt.show()

