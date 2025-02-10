import torch

x = torch.tensor([
    [1,2,3,4],
    [5,6,7,8]
])
print(x, x.shape)
x = torch.permute(x, (1,0))
print(x, x.shape)
x = torch.reshape(x, (4,2))
print(x, x.shape)
