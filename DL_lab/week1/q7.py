import torch
t1 = torch.rand(size=[2,3])
t2 = torch.rand(size=[2,3])
#7
print(t1, t1.device)
print(t2, t2.device)

t1 = t1.to(device="cuda")
print(t1, t1.device)
t2 = t2.to(device="cuda")
print(t2, t2.device)
t2 = torch.transpose(t2, 0, 1)

#8
ans = torch.matmul(t1, t2)
print(ans, ans.shape)

#9+10
print(ans[torch.argmax(ans)])
print(torch.argmax(ans))