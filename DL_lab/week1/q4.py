import torch
import numpy

b = torch.tensor([1,2,3,4])
print(b, type(b))
b = b.numpy()
print(b, type(b))
b = torch.from_numpy(b)
print(b, type(b))