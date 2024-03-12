import torch.nn as nn
import torch.nn.functional as F

# normal Resnet block
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channel)
        )

    def forward(self, X):
        Y = self.model(X)
        return F.relu(Y) + X