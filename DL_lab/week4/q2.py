import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np

loss_list = []
torch.manual_seed(42)

#Initialize inputs and outputs as per truth table of XOR
X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
#Y = torch.tensor([0, 1, 1, 0], dtype=torch.float32)
Y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        #self.w = torch.nn.Parameter(torch.rand([1]))
        #self.b = torch.nn.Parameter(torch.rand([1]))

        self.linear1 = nn.Linear(2, 2, bias=True)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(2,1,bias=True)
        self.activation2 = nn.ReLU()
        #self.activation2 = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.activation2(x)
        #x = self.activation2(x)
        return x

class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].to(device), self.Y[idx].to(device)

#create Dataset
full_dataset = MyDataset(X,Y)
batch_size = 1

#create dataloader
train_data_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#load the model to gpu
model = XORModel().to(device)
print(model)

#add the criterion which is the MSELoss
#loss_fn = torch.nn.MSELoss()
loss_fn = torch.nn.BCELoss()

#optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

epochs = 1000

#training an epoch
def train_one_epoch(epoch_index):
    totalloss = 0.
    #use enumerate(training_loader) instead of iter
    for i, data in enumerate(train_data_loader):
        #Every data input is an input + label pair
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        #Zero your gradients for every batch
        optimizer.zero_grad()

        #Make predictions for this batch
        outputs = model(inputs)

        #Compute loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        #adjust learning weights
        optimizer.step()

        #Gather data and report
        totalloss += loss.item()

    return totalloss/(len(train_data_loader)* batch_size)


for epoch in range(epochs):
    print(f'epoch {epoch}')

    #make sure model tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(epoch)
    loss_list.append(avg_loss)
    loss_list.append(avg_loss)
    print(f'Loss train {avg_loss}')
    if epoch % 1000 == 0:
        print(f"epoch {epoch}/{epochs}, loss:{avg_loss}")

#Model inference step
for param in model.named_parameters():
    print(param)
inputt = torch.tensor([0,1], dtype=torch.float32).to(device)
model.eval()
print(f"The input is ={inputt}")
print(f"Output y predicted = {model(inputt)}")
#display
plt.plot(loss_list)
plt.show()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of learnable parameters: {count_parameters(model)}")




















