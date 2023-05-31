import torch
from torch import nn


class CLIPLoss(nn.Module):
    def forward(self, advantage, ratio, epsilon):
        term_1 = -1.0 * advantage * ratio
        term_2 = -1.0 * advantage * torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        return torch.max(term_1, term_2).mean()


class VFLoss(nn.Module):
    def forward(self, target, estimate, clip: bool = False):
        if not clip:
            return 0.5 * ((target - estimate) ** 2).mean()
        # TODO: implement clipped version

