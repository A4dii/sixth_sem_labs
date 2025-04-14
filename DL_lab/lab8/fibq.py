import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fib(n):
    first = 0
    second = 1
    all_nums = [first, second]
    for i in range(n - 2):
        curr = first + second
        all_nums.append(curr)
        first = second
        second = curr
    return all_nums

x = fib(12)
seq_len = 3
n = len(x)
all_data = []
for i in range(n - seq_len):
    input_seq = x[i : i + seq_len]
    output_seq = x[i + seq_len]
    all_data.append([input_seq, output_seq])
X_data = []
y_data = []
for seq, target in all_data:
    X_data.append(seq)
    y_data.append(target)
X_data = np.array(X_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)
y_orig_min = np.min(y_data)
y_orig_max = np.max(y_data)
X_data = (X_data - np.min(X_data)) / (np.max(X_data) - np.min(X_data))
y_data = (y_data - y_orig_min) / (y_orig_max - y_orig_min)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.10, random_state=42, shuffle=False)

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32, device=device)
        self.y = torch.tensor(y, dtype=torch.float32, device=device)
        self.len = X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    def __len__(self):
        return self.len

train_dataset = MyDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=10, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(in_features=10, out_features=1)
    def forward(self, X):
        output, _ = self.rnn(X)
        output = output[:, -1, :]
        output = self.fc1(torch.relu(output))
        return output

model = RNNModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
epochs = 60000
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.view(-1, seq_len, 1)
        optimizer.zero_grad()
        y_pred = model(X_batch).reshape(-1)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 500 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

model.eval()
test_dataset = MyDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
predictions = []
true_values = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.view(-1, seq_len, 1)
        y_pred = model(X_batch)
        predictions.append(y_pred.item())
        true_values.append(y_batch.item())
predictions = np.array(predictions) * (y_orig_max - y_orig_min) + y_orig_min
true_values = np.array(true_values) * (y_orig_max - y_orig_min) + y_orig_min
print("Predictions:")
print(predictions)
print("\nTrue values:")
print(true_values)
plt.plot(true_values, label='True Values', marker='o')
plt.plot(predictions, label='Predicted Values', marker='x')
plt.legend()
plt.title('True vs Predicted Fibonacci Numbers')
plt.show()
