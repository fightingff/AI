import torch
import torch.nn as nn
from Resnet import ResBlock

# Conv2d (in_channel, out_channel, kernel_size, stride, padding)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # self.model = nn.Sequential(
        #     nn.BatchNorm2d(3),

        #     nn.Conv2d(3, 64, 5, 1, 2),
        #     nn.InstanceNorm2d(64),
        #     nn.ReLU(),                  # c5s1-64  64 * 64

        #     nn.Conv2d(64, 128, 3, 2, 1),
        #     nn.InstanceNorm2d(128),
        #     nn.ReLU(),                  # d128  32 * 32

        #     *[ResBlock(128, 128) for i in range(4)],     # R128 * 4 32 * 32

        #     nn.ConvTranspose2d(128, 64, 2, 2, 0),
        #     nn.InstanceNorm2d(64),     
        #     nn.ReLU(),                  # u64 64 * 64


        #     nn.Conv2d(64, 3, 5, 1, 2),
        #     nn.Tanh()
        # )
        # paper model too large 
        self.model = nn.Sequential(
            # nn.BatchNorm2d(3),

            nn.Conv2d(3, 64, 7, 1, 3),
            nn.InstanceNorm2d(64),
            nn.ReLU(),                  # c7s1-64  128 * 128

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),                  # d128  64 * 64

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),                  # d256  32 * 32

            *[ResBlock(256, 256) for i in range(6)],     # R256 * 6 32 * 32

            # nn.ConvTranspose2d(256, 128, 2, 2, 0),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.InstanceNorm2d(128),     
            nn.ReLU(),                  # u128 64 * 64

            # nn.ConvTranspose2d(128, 64, 2, 2, 0),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),      
            nn.ReLU(),                  # u64 128 * 128

            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()   
            # nn.Sigmoid()
        )
    
    def forward(self, X):
        return 0.5 * (self.model(X) + 1.0)