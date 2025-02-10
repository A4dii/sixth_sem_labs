import torch

torch.set_printoptions(precision=10)
x = torch.tensor(2.6, requires_grad=True)
y = torch.tensor(1.0, requires_grad=True)
z = torch.tensor(2.4, requires_grad=True)

def make_eq():
    p = 1 + (z*( (2*x)/(torch.sin(y)) ))
    q = torch.log(p)
    r = torch.tanh(q)
    return r

make_eq().backward()

print("Using grad:")
print("x-> ", x.grad)
print("y-> ", y.grad)
print("z-> ", z.grad)

a = 2*x
b = torch.sin(y)
c = a/b
d = c*z
e = torch.log(d+1)
f = torch.tanh(e)

print(a)
print(b)
print(c)
print(d)
print(e)
print(f)

print("Analytically:")
def analytical_soln():

    for_x =  (1-pow( (torch.tanh(e)),2)) * (1/(d+1)) * z * (1/b) *2
    for_y = (1-pow( torch.tanh(e),2) ) * (1/(d+1))  * z * (-(a/(b**2))) * torch.cos(y)
    for_z = (1-pow( (torch.tanh(e)),2)) * (1/(d+1)) * c
    print("x: ", for_x)
    print("y: ", for_y)
    print("z: ", for_z)
analytical_soln()

