import torch
x = torch.tensor([
    [1,2,3,4],
    [5,6,7,8]
])

print(x[0])
print(x[:, 1:])