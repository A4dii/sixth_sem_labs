import torch

b = torch.tensor(3.0)
x = torch.tensor(4.0)
w = torch.tensor(2.0, requires_grad=True)

def make_eq():
    u = w*x
    v = u+b
    a = torch.relu_(v)
    return a

make_eq().backward()

print("Using grad: ", w.grad)

relu_der =