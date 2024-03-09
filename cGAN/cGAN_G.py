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
            nn.Linear(1024, 7*7*128),
            nn.ReLU(),
            nn.BatchNorm1d(7*7*128),
            nn.Unflatten(1, (128, 7, 7))
        )
        self.embed = nn.Sequential(
            nn.Linear(10, 128),
            nn.Linear(128, 7*7),
            nn.Unflatten(1, (1, 7, 7))
        )
        self.model = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x, y):
        x = self.gen(x)
        y = self.embed(y)
        # x = torch.cat((x, y), 1)
        x = x + y
        return self.model(x)
    
    def loss_ls(self, scores_fake):
        return 0.5 * torch.mean((scores_fake - 1) ** 2)
    
    def loss_ce(self, scores_fake):
        return F.binary_cross_entropy_with_logits(scores_fake, torch.ones_like(scores_fake))