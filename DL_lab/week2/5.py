import torch
torch.set_printoptions(precision=10)
x = torch.tensor(2.0, requires_grad=True)
y = ( (8*pow(x,4)) + (3*pow(x,3)) + (7*pow(x,2)) + (6*x) + 3 )
y.backward()
print("Using grad: ", x.grad)

def analytical_soln():
    z = ( (32*pow(x,3)) + (9*pow(x,2)) + (14*x) + 6)
    return z

print("Analytical solution: ", analytical_soln())