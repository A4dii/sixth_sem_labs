import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn

# Load the dataset - adjust the file path as necessary
df = pd.read_csv("/content/drive/MyDrive/daily.csv")
df = df.dropna()

# Ensure the dataset has a 'Price' column (adjust if needed)
y = df['Price'].values
print("Total data points:", len(y))

# Normalize prices to [0, 1]
minm = y.min()
maxm = y.max()
y_normalized = (y - minm) / (maxm - minm)

# Create sequences: use last 10 days (Sequence_Length) to predict the 11th day
Sequence_Length = 10
X = []
Y = []
for i in range(len(y_normalized) - Sequence_Length):
    X.append(y_normalized[i:i+Sequence_Length])
    Y.append(y_normalized[i+Sequence_Length])

X = np.array(X)
Y = np.array(Y)

# Split data into training and testing sets (no shuffle to respect time series order)
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.10, random_state=42, shuffle=False
)
print("Train samples:", len(x_train), "Test samples:", len(x_test))

# Custom Dataset for Natural Gas Time Series
class NGTimeSeries(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.len = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len

# Create Dataset objects for train and test
train_dataset = NGTimeSeries(x_train, y_train)
test_dataset = NGTimeSeries(x_test, y_test)

# DataLoaders for training and testing
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64)

# Define the RNN model using LSTM for better long-term dependencies
class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=10, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        # Reshape input: [batch_size, Sequence_Length, 1]
        x = x.view(-1, Sequence_Length, 1)
        output, _ = self.lstm(x)
        # Use the output from the last time-step
        last_output = output[:, -1, :]
        output = self.fc1(torch.relu(last_output))
        return output

# Set device and initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNNModel().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs = 1500

# Training loop
for epoch in range(epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(x_batch).reshape(-1)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch} : Loss {loss.item():.6f}")

# Evaluate the model on the test set
model.eval()
test_predictions = []
test_targets = []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        y_pred = model(x_batch).view(-1)
        test_predictions.extend(y_pred.cpu().detach().numpy())
        test_targets.extend(y_batch.cpu().detach().numpy())

# Plot predicted vs. original normalized prices
plt.figure(figsize=(10, 5))
plt.plot(test_predictions, label='Predicted', marker='x')
plt.plot(test_targets, label='Original', marker='o')
plt.legend()
plt.title('Predicted vs. Original Normalized Prices')
plt.show()

# Denormalize the entire series for plotting
y_denorm = y_normalized * (maxm - minm) + minm
# Denormalize test predictions
test_pred_denorm = np.array(test_predictions) * (maxm - minm) + minm

# Plot the denormalized full series with test predictions overlayed
plt.figure(figsize=(10, 5))
plt.plot(y_denorm, label="Original Series")
start_index = len(y_denorm) - len(test_pred_denorm)
plt.plot(range(start_index, len(y_denorm)), test_pred_denorm, label="Predicted", marker='x')
plt.legend()
plt.title('Denormalized Series and Predictions')
plt.show()
