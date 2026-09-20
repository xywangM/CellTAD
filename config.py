#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import warnings

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
AUGMENTATION_DIR = os.path.join(PROJECT_ROOT, "augmentation")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "train")
IDENTIFICATION_DIR = os.path.join(PROJECT_ROOT, "identification")
VISUALIZATION_DIR = os.path.join(PROJECT_ROOT, "visualization")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

DEMO_DATA_DIR = os.path.join(DATA_DIR, "demo")

DEMO_DATASETS = {
    'GM-800U_006': {'dir': 'GM-800U_006', 'raw': 'pairs', 'chrom_sizes': None},
    'GSM4382149_cortex-p001-cb_001': {'dir': 'cortex_p001_c001', 'raw': 'hic',
                                      'chrom_sizes': 'data/chrom_mm10_sizes.txt'},
}
DEFAULT_DEMO_ANCHOR = 'GM-800U_006'


def demo_cell_chrom_dir(base_dir, anchor_cell, chrom):
    dataset = DEMO_DATASETS.get(anchor_cell, DEMO_DATASETS[DEFAULT_DEMO_ANCHOR])
    return os.path.join(base_dir, dataset['dir'], 'demo', chrom)


def get_default_config():
    cfg = {
        'base_dir': '/mnt/d/model/data/2025_scMicroc',
        'anchor_cell': 'GM-800U_006',
        'chrom': 'chr1',
        'resolution': 50000,
        'n_enhanced_candidates': 10,

        'hic_dir': None,
        'chrom_sizes_file': None,

        'output_dir': '/mnt/d/model/embedding/scMiroc/006/chr1',

        'wd': 100,

        'hidden_dim': 64, 'final_dim': 64, 'num_gcn_layers': 2,
        'dropout': 0.1, 'output_activation': 'none',

        'num_epochs': 220, 'warmup_epochs': 30, 'lr': 1e-3, 'weight_decay': 1e-4,
        'grad_clip': 5.0, 'device': 'auto',

        'adj_max_dist': 10, 'backbone_plus_one': 1.0,
        'edge_oe_clip': 2.0, 'feat_oe_clip': 20.0,
        'agg_norm': 'sym', 'use_batchnorm': False, 'use_ice': False,

        'n_aug_views': 0, 'aug_sigma': 0.02,
        'n_pseudobulk_tads': 96, 'tad_insul_window': 5, 'min_tad_size': 3,

        'alpha_max': 0.7, 'alpha_ramp_epochs': 10,
        'ncla_tau': 0.2, 'ncla_sample': 512, 'ncla_r_local': 2, 'ncla_neg_samples': 128,
        'K_GIC': 40, 'gic_tau': 0.35, 'gic_z_scale_init': 20.0,

        'checkpoint_metric': 'ncla',
        'ckpt_smooth_evals': 3,
        'adj_std_guard_ratio': 1.10,
        'ord_guard_ratio': 0.98,

        'aggregate': 'concat', 'eval_freq': 5, 'eval_start': 30,

        'save_znorm': True,
        'run_oversmoothing_diagnostic': True,

        'run_augmentation_pipeline': False,
        'augmentation_force': False,
        'augmentation_start_from': 1,
        'augmentation_only': None,
        'augmentation_notebook_cwd': None,

        'demo_layout': False,

        'run_identification': False,
        'run_visualization': False,
        'visualization_outdir': '/mnt/d/model/output/visualization/tad',
        'visualization_regions': [(33, 36), (66, 69), (85, 88), (90, 93)],
    }
    return cfg


def apply_demo_mode(cfg):
    if cfg.get('anchor_cell') not in DEMO_DATASETS:
        cfg['anchor_cell'] = DEFAULT_DEMO_ANCHOR
    cfg['base_dir'] = DEMO_DATA_DIR
    cfg['chrom'] = 'chr1'
    cfg['resolution'] = 50000
    cfg['demo_layout'] = True

    cell_chrom_dir = demo_cell_chrom_dir(DEMO_DATA_DIR, cfg['anchor_cell'], cfg['chrom'])
    cfg['output_dir'] = os.path.join(cell_chrom_dir, 'output', 'embedding')
    cfg['visualization_outdir'] = os.path.join(cell_chrom_dir, 'output', 'visualization', 'tad')
    return cfg


def cell_dir_name(name):
    """cell_XXX directory name of a cell; a trailing number is zero-padded to 3 digits like Steps 6-7 do."""
    m = re.search(r'(\d+)$', str(name))
    return f"cell_{int(m.group(1)):03d}" if m else f"cell_{str(name).split('_')[-1]}"


def read_cell_map(base_dir):
    """Read cell_id_map.tsv into [(cell_id, original_id)]."""
    path = os.path.join(base_dir, 'cell_id_map.tsv')
    rows = []
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                p = line.rstrip('\n').split('\t')
                if len(p) >= 2 and p[0] != 'cell_id':
                    rows.append((p[0], p[1]))
    return rows


def resolve_cell_id(base_dir, anchor_cell):
    """Map an original name or cell_XXX to cell_XXX; None when the map does not exist yet."""
    rows = read_cell_map(base_dir)
    for cell_id, original_id in rows:
        if anchor_cell in (cell_id, original_id):
            return cell_id
    if rows:
        raise ValueError(f"anchor_cell '{anchor_cell}' is not in {os.path.join(base_dir, 'cell_id_map.tsv')} "
                         f"(examples: {[r[1] for r in rows[:3]]})")
    return None


def resolve_paths(cfg):
    base_dir = cfg['base_dir']
    anchor_cell = cfg['anchor_cell']
    chrom = cfg['chrom']

    if cfg.get('demo_layout', False):
        cell_chrom_dir = demo_cell_chrom_dir(base_dir, anchor_cell, chrom)
        raw_kind = DEMO_DATASETS.get(anchor_cell, DEMO_DATASETS[DEFAULT_DEMO_ANCHOR])['raw']

        if raw_kind == 'hic':
            cfg['anchor_path'] = os.path.join(cell_chrom_dir, 'raw', f'{anchor_cell}.contacts.hic')
            cfg['anchor_cool_path'] = None
        else:
            cfg['anchor_path'] = os.path.join(
                cell_chrom_dir, 'raw', f'{anchor_cell}_allValidPairs.txt')
            cfg['anchor_cool_path'] = os.path.join(
                cell_chrom_dir, 'raw', f'{anchor_cell}.cool')

        cfg['enhanced_paths'] = [
            os.path.join(cell_chrom_dir, 'augmented', f'enhanced_top{i}.cool')
            for i in range(1, cfg['n_enhanced_candidates'] + 1)
        ]

        cfg['cell_ids_in_order_path'] = os.path.join(
            cell_chrom_dir, 'embedding', 'schicluster_embedding', 'cell_ids_in_order.txt')
        cfg['total_decomp_path'] = os.path.join(
            cell_chrom_dir, 'embedding', 'schicluster_embedding', 'total_decomp.npz')

        cfg['pseudobulk_path'] = os.path.join(
            cell_chrom_dir, 'pseudobulk', f'{chrom}_bulk_{cfg["resolution"]}bp.npy')
        cfg['pseudobulk_tads_path'] = os.path.join(
            cell_chrom_dir, 'pseudobulk', f'{chrom}_TADs.csv')

    else:
        if cfg.get('hic_dir'):
            file_id = resolve_cell_id(base_dir, anchor_cell)
            if file_id is None:
                file_id = anchor_cell if re.fullmatch(r'cell_\d+', anchor_cell) else cell_dir_name(anchor_cell)
                warnings.warn(f"cell_id_map.tsv not found in {base_dir}; assuming '{anchor_cell}' -> '{file_id}' "
                              f"until the .hic conversion (Step 0) has run")
            enhanced_dir = file_id
        else:
            file_id = anchor_cell
            enhanced_dir = cell_dir_name(anchor_cell)
        cfg['cell_id'] = file_id

        cfg['anchor_path'] = os.path.join(
            base_dir, 'GSE279583_extracted', f'{file_id}.allValidPairs.txt')
        cfg['anchor_cool_path'] = os.path.join(
            base_dir, 'schicluster_cool', f'{file_id}.cool')

        cfg['enhanced_paths'] = [
            os.path.join(base_dir, 'enhanced_maps', enhanced_dir, chrom,
                          f'enhanced_top{i}.cool')
            for i in range(1, cfg['n_enhanced_candidates'] + 1)
        ]

        cfg['cell_ids_in_order_path'] = os.path.join(
            base_dir, 'embedding', '800U', 'cell_ids_in_order.txt')
        cfg['total_decomp_path'] = os.path.join(
            base_dir, 'embedding', '800U', 'decomp', 'total_decomp.npz')

        cfg['pseudobulk_tads_path'] = os.path.join(
            base_dir, 'tadgate_results', 'tad_boundaries', f'{chrom}_TADs.csv')
        cfg['pseudobulk_path'] = None

    return cfg