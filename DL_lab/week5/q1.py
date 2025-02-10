import torch
import torch.nn.functional as F
image = torch.rand(6,6)
print("image=", image)
#Add a new dimension along 0th dimension
#i.e. (6,6) becomes (1,6,6). This is because
#pytorch expects the input to conv2D as 4d tensor
image = image.unsqueeze(dim=0)
print("image.shape=", image.shape)
image = image.unsqueeze(dim=0)
print("image.shape=", image.shape)
print("image=", image)
kernel = torch.ones(3,3)
#kernel = torch.rand(3,3)
print("kernel=", kernel)
kernel = kernel.unsqueeze(dim=0)
kernel = kernel.unsqueeze(dim=0)
#Perform the convolution
outimage = F.conv2d(image, kernel, stride=1, padding=0)
print("outimage=", outimage, outimage.shape)

print("\n--------------------Padding check--------------------\n")

for i in range(1, 10):
    print("padding= ", i)
    out_temp = F.conv2d(image, kernel, stride=1, padding=i)
    print("out= ", out_temp.shape)

print("\n--------------------Stride check--------------------\n")
for i in range(1, 10):
    print("stride= ", i)
    out_temp = F.conv2d(image, kernel, stride=i, padding=0)
    print("out= ", out_temp.shape)

print("\n--------------------Padding + Stride check--------------------\n")

for i in range(2, 10):
    print("stride= ", i)
    out_temp = F.conv2d(image, kernel, stride=i, padding=i)
    print("out= ", out_temp.shape)

print("\n--------------------Kernel check--------------------\n")

for i in range(2, 7):
    kernel = torch.ones(i, i)
    kernel = kernel.unsqueeze(dim=0)
    kernel = kernel.unsqueeze(dim=0)
    print("kernel_size= ", i, i)
    out_temp = F.conv2d(image, kernel, stride=1, padding=0)
    print("out= ", out_temp.shape)