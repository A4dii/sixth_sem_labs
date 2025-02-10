import torch

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0)

def make_eq():
    x = 2*a + 3*b
    y = 5*a*a + 3*b*b*b
    z = 2*x + 3*y
    return z

def f():
    z = 2*2 + 3*10*a
    return z

make_eq().backward()

print("Using grad: ",a.grad)
print("Analytical: ",f())