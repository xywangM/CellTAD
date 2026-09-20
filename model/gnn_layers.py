#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


class PaperGCNLayer(nn.Module):
    """Single GCN layer: h_i' = ELU(sum_j A_ij W h_j), with edge weights w taken from the graph."""

    def __init__(self, in_dim, out_dim, dropout=0.1, act=True, use_bn=False):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=True)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)
        self.dropout = nn.Dropout(dropout)
        self.act = act
        self.bn = nn.BatchNorm1d(out_dim) if use_bn else None

    def forward(self, g, h):
        h = self.dropout(h)
        with g.local_scope():
            g.ndata['Wh'] = self.W(h)
            g.apply_edges(lambda e: {'m': e.data['w'].unsqueeze(-1) * e.src['Wh']})
            g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'h_sum'))
            h_out = g.ndata['h_sum']
        if self.bn is not None:
            h_out = self.bn(h_out)
        if self.act:
            h_out = F.elu(h_out)
        return h_out