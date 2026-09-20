#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn_layers import PaperGCNLayer


class PaperGCN(nn.Module):
    """Paper-style shared GNN encoder: multiple layers of ELU(sum_j A_hat_ij W h_j)."""

    def __init__(self, in_dim, hidden_dim=64, final_dim=64,
                 num_layers=2, dropout=0.1, output_activation='none', use_bn=False):
        super().__init__()
        self.output_activation = output_activation
        self.layers = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [final_dim]
        for li in range(num_layers):
            self.layers.append(
                PaperGCNLayer(dims[li], dims[li + 1], dropout,
                              act=(li < num_layers - 1), use_bn=use_bn))

    def forward(self, g, x):
        h = x
        for layer in self.layers:
            h = layer(g, h)
        if self.output_activation == 'tanh':
            h = torch.tanh(h)
        elif self.output_activation == 'l2norm':
            h = F.normalize(h, p=2, dim=1)
        return h


@torch.no_grad()
def layerwise_similarity_report(encoder: "PaperGCN", g, x, n_pairs=2000, seed=42):
    """Print per-layer cosine similarity statistics of node representations (over-smoothing diagnostic)."""
    gen = torch.Generator(device='cpu').manual_seed(seed)
    N = x.shape[0]
    idx_a = torch.randint(0, N, (n_pairs,), generator=gen)
    idx_b = torch.randint(0, N, (n_pairs,), generator=gen)

    print("=" * 70)
    print("[diagnostic: over-smoothing] per-layer node representation similarity "
          "(pure forward pass, no training/backprop)")
    print("  A value that rises with depth and approaches 1 means the GCN is making "
          "different nodes increasingly similar --")
    print("  a candidate independent cause of chain effects (as opposed to the TAD structure itself).")
    print("  If the input layer is already high, the issue is in the features, not the GCN depth.")
    print("=" * 70)

    h = x
    layers = [("input", h)]
    for li, layer in enumerate(encoder.layers):
        h = layer(g, h)
        layers.append((f"layer{li+1}", h))

    for name, h_l in layers:
        h_cpu = h_l.detach().cpu()
        norm = h_cpu / (h_cpu.norm(dim=1, keepdim=True) + 1e-8)
        a = norm[idx_a]
        b = norm[idx_b]
        sims = (a * b).sum(dim=1)
        print(f"  {name:>8}: mean_cos={sims.mean().item():.4f}  "
              f"p50={sims.median().item():.4f}  "
              f"p90={sims.kthvalue(int(0.9 * len(sims))).values.item():.4f}  "
              f"std={sims.std().item():.4f}")
    print("=" * 70)