# Generator of cGAN

# Generator of GAN

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Generator(nn.Module):
    def __init__(self, noise_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(noise_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 16*16*64),
            nn.ReLU(),
            nn.BatchNorm1d(16*16*64),
            nn.Unflatten(1, (64, 16, 16))
        )
        self.embed = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=2, stride=2),
            nn.LeakyReLU(0.2),

            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            # nn.MaxPool2d(kernel_size=2, stride=2), # 16 * 16
            nn.Conv2d(64, 64, kernel_size=2, stride=2),
            nn.LeakyReLU(0.2),
        )
        self.model = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        x = self.gen(x)
        y = self.embed(y)
        x = torch.cat((x, y), 1)
        return self.model(x)
    
    def loss_l1(self, scores_fake):
        return torch.mean(torch.abs(scores_fake - 1))

    def loss_ls(self, scores_fake):
        return 0.5 * torch.mean((scores_fake - 1) ** 2)
    
    def loss_ce(self, scores_fake):
        return F.binary_cross_entropy_with_logits(scores_fake, torch.ones_like(scores_fake))