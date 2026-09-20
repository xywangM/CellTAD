#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn

from .encoder import PaperGCN
from .gic import GICModule


class CellTADModel(nn.Module):
    """Encoder (PaperGCN) plus GIC module (GICModule)."""

    def __init__(self, in_dim, hidden_dim=64, final_dim=64, num_gcn_layers=2,
                 dropout=0.1, output_activation='none', use_batchnorm=False,
                 K_gic=40, gic_tau=1.0, gic_z_scale_init=1.0):
        super().__init__()
        self.encoder = PaperGCN(
            in_dim=in_dim, hidden_dim=hidden_dim, final_dim=final_dim,
            num_layers=num_gcn_layers, dropout=dropout,
            output_activation=output_activation, use_bn=use_batchnorm,
        )
        self.gic = GICModule(final_dim, K_gic, tau=gic_tau, z_scale_init=gic_z_scale_init)

    def encode(self, g, x):
        """Encode one view's features x on graph g into node embeddings."""
        return self.encoder(g, x)

    def forward(self, g, x):
        return self.encode(g, x)