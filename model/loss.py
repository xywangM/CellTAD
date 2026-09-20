#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch


def combine_losses(L_ncla, L_gic, alpha, gic_active, device=None):
    """Return (total loss, L_gic); total = alpha * L_ncla + (1 - alpha) * L_gic when gic_active, else L_ncla."""
    if gic_active:
        if L_gic is None:
            raise ValueError("L_gic must be provided when gic_active=True")
        total = alpha * L_ncla + (1.0 - alpha) * L_gic
        return total, L_gic

    if L_gic is None:
        L_gic = torch.tensor(0.0, device=device if device is not None else L_ncla.device)
    return L_ncla, L_gic