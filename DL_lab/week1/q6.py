import torch
#q5 + q6
x = torch.rand(size=(7,7))
y = torch.rand(size=(1,7))
y = torch.transpose(y, 0,1)

print("First:", x, x.shape)
print("Second:",y, y.shape, y.dtype)

f = torch.matmul(x, y)
print(f, f.shape)

