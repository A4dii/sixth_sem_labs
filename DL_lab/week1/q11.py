import torch 
torch.manual_seed(7)
t1 = torch.rand(size=(1,1,1,10))
print(t1, t1.shape)
t1=t1.squeeze()
print(t1, t1.shape)
