import torch
from matplotlib import pyplot as plt

def analytical_soln(y_p, y, x):
    xGrad = 2 * (y_p-y) * x
    bGrad = 2 * (y_p - y)
    return xGrad, bGrad

x = torch.tensor([2 ,4])
y = torch.tensor([20, 40])

b = torch.rand([1], requires_grad=True)
w = torch.rand([1], requires_grad=True)
print(f"The parameters are {w}, and {b}")

learning_rate = torch.tensor(0.001)

loss_list = []

for epochs in range(2):
    print(f"Epoch{epochs}")
    loss = 0.0
    for j in range(len(x)):
        a = w * x[j]
        y_p = a + b
        loss += (y_p - y[j])**2
        wGrad, bGrad = analytical_soln(y_p, y[j], x[j])
        print(f"Analyticaly--w_grad:{wGrad}, b_grad:{bGrad}")
    loss = loss / len(x)
    loss_list.append(loss.item())
    #dL/dw and dL/db
    loss.backward()

    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    print(f"w_grad: {w.grad}, b_grad:{b.grad}")
    w.grad.zero_()
    b.grad.zero_()
    print(f"The parameters are w={w}, b={b}, and loss={loss.item()}")

plt.plot(loss_list)
plt.show()
