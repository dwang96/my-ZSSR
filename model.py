import torch 
import torch.nn as nn
from math import sqrt

class ZSSRnet(nn.Module):
    def __init__(self, input_channels=3, channels=128, kernel_size=3, num_layers=8, device='cuda'):
        super(ZSSRnet, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels = input_channels, out_channels = channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.conv2 = nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.conv3 = nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.conv4 = nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.conv5 = nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.conv6 = nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.conv7 = nn.Conv2d(in_channels = channels, out_channels = channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.conv8 = nn.Conv2d(in_channels = channels, out_channels = input_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.device = device
        self.to(device)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

        
    def forward(self, residual_in):
        
        inputs = self.conv1(self.relu(residual_in))
        x = inputs
        x = self.conv2(self.relu(x))
        x = self.conv3(self.relu(x))
        x = self.conv4(self.relu(x))
        x = self.conv5(self.relu(x))
        x = self.conv6(self.relu(x))
        x = self.conv7(self.relu(x))
        out = self.conv8(self.relu(x))
        
        return out + residual_in