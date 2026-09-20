#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from pathlib import Path

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model.model import CellTADModel
from train.dataset import CellTADDataset
from train.trainer import Trainer


def _default_config():
    """Fallback config source for running this file standalone (not via CellTAD.py)."""
    from config import get_default_config, resolve_paths
    cfg = get_default_config()
    cfg = resolve_paths(cfg)
    if cfg['device'] == 'auto':
        cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    return cfg


def main(cfg=None):
    if cfg is None:
        cfg = _default_config()

    device_name = cfg['device']
    if device_name == 'auto':
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    dev = torch.device(device_name)
    odir = Path(cfg['output_dir'])
    odir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("CellTAD embedding training (train/ framework, paper-aligned logic)")
    print(f"  Views: anchor + {cfg['n_aug_views'] if cfg['n_aug_views'] > 0 else len(cfg['enhanced_paths'])} augmented maps")
    print(f"  adj_max_dist={cfg['adj_max_dist']}, edge_oe_clip={cfg['edge_oe_clip']}, "
          f"backbone_plus_one={cfg['backbone_plus_one']}, "
          f"alpha_max={cfg.get('alpha_max')}, gic_tau={cfg.get('gic_tau')} "
          f"-- check the full [diag-clip] printout in the training log")
    print("=" * 78)

    dataset = CellTADDataset(
        dev,
        anchor_path=cfg['anchor_path'], enhanced_paths=cfg['enhanced_paths'],
        chrom=cfg['chrom'], resolution=cfg['resolution'], wd=cfg['wd'],
        use_ice=cfg['use_ice'], feat_oe_clip=cfg['feat_oe_clip'],
        n_aug_views=cfg['n_aug_views'], aug_sigma=cfg['aug_sigma'],
        tad_insul_window=cfg['tad_insul_window'],
        n_pseudobulk_tads=cfg['n_pseudobulk_tads'], min_tad_size=cfg['min_tad_size'],
        backbone_plus_one=cfg['backbone_plus_one'], edge_oe_clip=cfg['edge_oe_clip'],
        adj_max_dist=cfg['adj_max_dist'], agg_norm=cfg.get('agg_norm', 'sym'),
    )

    model = CellTADModel(
        in_dim=dataset.fd, hidden_dim=cfg['hidden_dim'], final_dim=cfg['final_dim'],
        num_gcn_layers=cfg['num_gcn_layers'], dropout=cfg['dropout'],
        output_activation=cfg['output_activation'], use_batchnorm=cfg.get('use_batchnorm', False),
        K_gic=cfg['K_GIC'], gic_tau=cfg.get('gic_tau', 1.0),
        gic_z_scale_init=cfg.get('gic_z_scale_init', 1.0)).to(dev)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"  Model: {cfg['num_gcn_layers']}-layer paper-GCN + GIC | {n_param:,} params")

    trainer = Trainer(
        model, dev,
        num_epochs=cfg['num_epochs'], warmup_epochs=cfg['warmup_epochs'],
        lr=cfg['lr'], weight_decay=cfg['weight_decay'], grad_clip=cfg['grad_clip'],
        checkpoint_metric=cfg.get('checkpoint_metric', 'ncla'),
        ckpt_smooth_evals=cfg.get('ckpt_smooth_evals', 1),
        alpha_max=cfg.get('alpha_max', cfg.get('alpha', 0.2)),
        alpha_ramp_epochs=cfg.get('alpha_ramp_epochs', 0),
        adj_std_guard_ratio=cfg.get('adj_std_guard_ratio', 0),
        ord_guard_ratio=cfg.get('ord_guard_ratio', 0),
        ncla_tau=cfg['ncla_tau'], ncla_sample=cfg['ncla_sample'],
        ncla_r_local=cfg['ncla_r_local'], ncla_neg_samples=cfg['ncla_neg_samples'],
        K_GIC=cfg['K_GIC'], gic_tau=cfg.get('gic_tau', 1.0),
        gic_z_scale_init=cfg.get('gic_z_scale_init', 1.0),
        eval_start=cfg['eval_start'], eval_freq=cfg['eval_freq'],
        aggregate=cfg['aggregate'],
    )
    logs = trainer.fit(dataset)
    final, view_embs = trainer.get_final_embedding(dataset)

    if cfg.get('run_oversmoothing_diagnostic', False):
        try:
            from model.encoder import layerwise_similarity_report
            with torch.no_grad():
                layerwise_similarity_report(model.encoder, dataset.g, dataset.feats[0])
        except Exception as ex:
            print(f"  Over-smoothing diagnostic skipped (does not affect training/saving): {ex}")

    pre = f"{cfg['anchor_cell']}_{cfg['chrom']}"
    np.save(str(odir / f'{pre}_embedding.npy'), final)
    if cfg.get('save_znorm', True):
        zn = final / (np.linalg.norm(final, axis=1, keepdims=True) + 1e-8)
        np.save(str(odir / f'{pre}_embedding_znorm.npy'), zn.astype(np.float32))
    np.save(str(odir / f'{pre}_insulation_scores.npy'), dataset.insulation_scores)
    if dataset.tad_labels is not None:
        np.save(str(odir / f'{pre}_pseudobulk_tad_labels.npy'), dataset.tad_labels)
        np.save(str(odir / f'{pre}_pseudobulk_tad_boundaries.npy'), np.array(dataset.tad_boundaries))
    for i, e in enumerate(view_embs):
        np.save(str(odir / f'{pre}_view{i}_embedding.npy'), e)
    with open(str(odir / f'{pre}_losses_v62.json'), 'w') as f:
        json.dump(logs, f, indent=2)
    torch.save({'model': model.state_dict(), 'config': cfg, 'N': dataset.N, 'fd': dataset.fd},
               str(odir / f'{pre}_model_v62.pth'))

    print(f"  Saved to {odir}")
    print("Training complete.")
    return model, logs, final


if __name__ == "__main__":
    main()