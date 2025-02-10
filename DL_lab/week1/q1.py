import torch

t1 = torch.tensor([
            [6,7,8,9],
            [10,11,12,13],
            [14,15,16,17]
])

t2 = torch.tensor([
            [1,2,3,4],
            [20,21,22,23],
            [24,25,26,27]
])

#Q1
#Reshape
t3 = torch.tensor([1,2,3,4])
t3 =torch.reshape(t3, (2,2))
#Viewing
print(t3)
#stacking
f = torch.stack((t1, t2), dim=1)
print(f, f.shape)
#squeezing
t4 = torch.zeros(2,1,2,1,2)
print(t4, t4.shape)
t4 = torch.squeeze(t4)
print(t4, t4.shape)
#unsqueezing
t5 = torch.tensor([[1], [2]])
print(t5, t5.shape)
t5 = torch.squeeze(t5)
print(t5, t5.size)
t5 = torch.unsqueeze(t5, dim=1)
print(t5, t5.shape)
