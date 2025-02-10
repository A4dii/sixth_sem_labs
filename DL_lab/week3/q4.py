import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

x_data = torch.tensor([5.0, 7.0, 12.0, 16.0, 20.0]).view(-1, 1)
y_data = torch.tensor([40.0, 120.0, 180.0, 210.0, 240.0]).view(-1, 1)

class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

learning_rate = 0.001
epochs = 100
batch_size = 2

dataset = RegressionDataset(x_data, y_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = RegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

loss_list = []
for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_x, batch_y in dataloader:
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    loss_list.append(avg_loss)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

plt.plot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

for name, param in model.named_parameters():
    print(f"{name}: {param.data}")
