# Discriminator of cGAN

import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.BatchNorm2d(6),
            nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=2, stride=2),
            nn.LeakyReLU(0.2),

            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=2, stride=2),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(64*16*16, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
        )
    
    def forward(self, x, y):
        x = torch.cat((x, y), 1)
        return self.model(x)
    
    def loss_l1(self, scores_real, scores_fake):
        return (torch.mean(torch.abs(scores_real - 0.9)) + torch.mean(torch.abs(scores_fake)))

    def loss_ls(self, scores_real, scores_fake):
        return 0.5 * (torch.mean((scores_real - 0.9) ** 2) + torch.mean(scores_fake ** 2))
    
    def loss_ce(self, scores_real, scores_fake):
        return F.binary_cross_entropy_with_logits(scores_real, torch.ones_like(scores_real) * 0.9) + F.binary_cross_entropy_with_logits(scores_fake, torch.zeros_like(scores_fake))