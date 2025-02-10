import torch
torch.set_printoptions(precision=10)
x = torch.tensor(2.0, requires_grad=True)
f = torch.exp(-(x*x) - (2*x) - torch.sin(x))
f.backward()
print("Using grad: ", x.grad)

def analytical_soln():
    z = (-(2*x) - 2 - torch.cos(x)) * torch.exp(-(x*x) - (2*x) - torch.sin(x))
    return z

print("Analytical solution: ",analytical_soln())