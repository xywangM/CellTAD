#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


def loss_ncla_paper(z_anchor, z_views, tau=0.2, sample=512, r_local=2, n_neg=128):
    N = z_anchor.shape[0]
    dev = z_anchor.device
    za = F.normalize(z_anchor, p=2, dim=1)
    zvs = [F.normalize(zv, p=2, dim=1) for zv in z_views]

    n_anchor = min(sample, N)
    anchors = torch.randperm(N, device=dev)[:n_anchor].tolist()

    losses = []
    for a in anchors:
        q = za[a]
        pos_list = [zv[a] for zv in zvs]
        local_js = []
        for off in range(-r_local, r_local + 1):
            j = a + off
            if j == a or j < 0 or j >= N:
                continue
            pos_list.append(za[j]); local_js.append(j)
        if len(pos_list) == 0:
            continue
        pos = torch.stack(pos_list)

        forbidden = torch.tensor(local_js + [a], device=dev, dtype=torch.long)
        cand = torch.randint(0, N, (n_neg * 2,), device=dev)
        keep = ~torch.isin(cand, forbidden)
        neg_idx = cand[keep][:n_neg]
        if neg_idx.numel() == 0:
            continue
        neg = za[neg_idx]

        sim_pos = (pos * q.unsqueeze(0)).sum(dim=1) / tau
        sim_neg = (neg * q.unsqueeze(0)).sum(dim=1) / tau
        log_num = torch.logsumexp(sim_pos, dim=0)
        log_den = torch.logsumexp(torch.cat([sim_pos, sim_neg]), dim=0)
        losses.append(-(log_num - log_den))

    if len(losses) == 0:
        return torch.tensor(0.0, device=dev, requires_grad=True)
    return torch.stack(losses).mean()