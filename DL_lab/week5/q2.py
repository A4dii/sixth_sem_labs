import torch
import torch.nn.functional as F

image = torch.rand(6, 6)
print("image=", image)

image = image.unsqueeze(dim=0)
image = image.unsqueeze(dim=0)
print("image.shape=", image.shape)

out_channels = 3

kernel = torch.ones(out_channels, 1, 3, 3)  # Shape: (3, 1, 3, 3)
print("kernel=", kernel, kernel.shape)

outimage = F.conv2d(image, kernel, stride=1, padding=0)
print("outimage=", outimage, outimage.shape)

m = torch.nn.Conv2d(1, 3, (3, 3), stride=1, padding=0, bias=False)
m.weight.data = kernel
outimage2 = m(image)

print("outimage2=", outimage2, outimage2.shape)