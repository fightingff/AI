import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # self.model = nn.Sequential(
        #     nn.BatchNorm2d(3),

        #     nn.Conv2d(3, 64, 4, 2, 1),
        #     nn.LeakyReLU(0.2),          # C64 32 * 32

            
        #     nn.Conv2d(64, 128, 4, 2, 1),
        #     nn.InstanceNorm2d(128),
        #     nn.LeakyReLU(0.2),          # C128 16 * 16


        #     nn.Conv2d(128, 256, 4, 2, 1),
        #     nn.InstanceNorm2d(256),
        #     nn.LeakyReLU(0.2),          # C256 8 * 8

        #     nn.Conv2d(256, 128, 4, 2, 1),
        #     nn.InstanceNorm2d(128),
        #     nn.LeakyReLU(0.2),          # C128 4 * 4

        #     nn.Flatten(),
        #     nn.Linear(128*4*4, 256),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(256, 1)          # get answer
        # )

        # the paper model is too large
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),          # C64 64 * 64

            
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),          # C128 32 * 32


            nn.Conv2d(128, 256, 4, 2, 1),
            nn.Dropout(0.2),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),          # C256 16 * 16

            nn.Conv2d(256, 512, 4, 1, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),          # C512 16 * 16

            nn.Conv2d(512, 1, 4, 1, 1),
            # nn.ReLU(),
            nn.AvgPool2d(14, 14, 0),      # 14 * 14
            nn.Flatten()
        )
    
    def forward(self, X):
        return self.model(X)
