#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn


class Discriminator(nn.Module):
    """Dot-product score between a node representation h and a cluster summary z. No parameters."""

    def forward(self, h, z):
        return (h * z).sum(dim=1)