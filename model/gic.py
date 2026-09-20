#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from .discriminator import Discriminator


class GICModule(nn.Module):
    def __init__(self, dim, K, tau=1.0, z_scale_init=1.0):
        super().__init__()
        self.K = K
        self.tau = tau
        self.centroids = nn.Parameter(torch.randn(K, dim) * 0.1)
        self.z_scale = nn.Parameter(torch.tensor(float(z_scale_init)))
        self.initialized = False
        self.discriminator = Discriminator()

    @torch.no_grad()
    def init_centroids(self, h):
        km = KMeans(self.K, n_init=10, random_state=42)
        km.fit(h.detach().cpu().numpy())
        self.centroids.data = torch.from_numpy(km.cluster_centers_).float().to(h.device)
        self.initialized = True

    def soft_assign(self, h):
        dist2 = torch.cdist(h, self.centroids).pow(2)
        mean = dist2.mean(dim=1, keepdim=True)
        std = dist2.std(dim=1, keepdim=True).clamp(min=1e-8)
        dist2_z = (dist2 - mean) / std
        return F.softmax(-dist2_z / self.tau, dim=1)

    def forward(self, h_real, h_corrupt):
        r = self.soft_assign(h_real)
        z = torch.sigmoid(self.z_scale * (r @ self.centroids))
        pos = self.discriminator(h_real, z)
        neg = self.discriminator(h_corrupt, z)
        return -(F.logsigmoid(pos).mean() + F.logsigmoid(-neg).mean())

    @torch.no_grad()
    def diagnostics(self, h):
        r = self.soft_assign(h)
        raw_logits = r @ self.centroids
        scaled_logits = self.z_scale * raw_logits
        z = torch.sigmoid(scaled_logits)
        z_centroid_only = torch.sigmoid(self.z_scale * self.centroids)

        return {
            'h_mean': h.mean().item(), 'h_std': h.std().item(),
            'h_min': h.min().item(), 'h_max': h.max().item(),
            'c_mean': self.centroids.mean().item(), 'c_std': self.centroids.std().item(),
            'c_min': self.centroids.min().item(), 'c_max': self.centroids.max().item(),
            'z_scale': self.z_scale.item(),
            'logit_mean': scaled_logits.mean().item(), 'logit_std': scaled_logits.std().item(),
            'logit_min': scaled_logits.min().item(), 'logit_max': scaled_logits.max().item(),
            'z_std_over_nodes': z.std(dim=0).mean().item(),
            'sigmoid_saturation_ratio': ((z < 0.02) | (z > 0.98)).float().mean().item(),
            'z_std_over_clusters': z_centroid_only.std(dim=0).mean().item(),
            'r_max_mean': r.max(dim=1).values.mean().item(),
        }