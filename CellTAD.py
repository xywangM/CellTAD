#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse

main_dir = os.path.dirname(os.path.abspath(__file__))
if main_dir not in sys.path:
    sys.path.insert(0, main_dir)

from config import get_default_config, apply_demo_mode, resolve_paths, DEMO_DATASETS
from logger import Logger


def create_parser():
    p = argparse.ArgumentParser(description="CellTAD: single-cell Hi-C TAD embedding & detection")

    p.add_argument('--demo', action='store_true',
                    help='Switch to the small demo dataset bundled with the repo (data/demo/). '
                         'The demo is chosen by --anchor-cell (default GM-800U_006)')
    p.add_argument('--base-dir', type=str, default=None,
                    help='Dataset root directory (formerly BASE_DIR); uses config.py default if omitted')
    p.add_argument('--anchor-cell', type=str, default=None, help='Anchor cell name')
    p.add_argument('--chrom', type=str, default=None, help='Chromosome, e.g. chr1')
    p.add_argument('--resolution', type=int, default=None, help='Resolution (bp)')
    p.add_argument('--output-dir', type=str, default=None, help='Output directory')
    p.add_argument('--visualization-outdir', type=str, default=None,
                    help='Where visualization/tad.py writes figures and TAD tables; '
                         'defaults to <output-dir>/visualization/tad when --output-dir is given, otherwise config.py')
    p.add_argument('--hic-dir', type=str, default=None,
                    help='Directory with single-cell *.hic files (.hic input). Omit when the data already is pairs. '
                         'Use together with --run-augmentation: Step 0 converts .hic to pairs')
    p.add_argument('--chrom-sizes', type=str, default=None,
                    help='Chromosome size table: data/chrom_hg38_sizes.txt (default) or data/chrom_mm10_sizes.txt; '
                         'a relative path is looked up in the current directory, then in the project root')

    p.add_argument('--num-epochs', type=int, default=None)
    p.add_argument('--device', type=str, default=None, choices=['auto', 'cpu', 'cuda'])

    p.add_argument('--run-augmentation', action='store_true',
                    help='Run the augmentation/ seven-step pipeline (validate/generate augmented maps) before training')
    p.add_argument('--augmentation-force', action='store_true',
                    help='Ignore existing augmentation outputs and force a rerun')
    p.add_argument('--augmentation-start-from', type=int, default=None)
    p.add_argument('--augmentation-only', type=int, default=None)
    p.add_argument('--augmentation-anchor-only', action='store_true',
                    help='Generate augmented maps (Steps 6-7) only for --anchor-cell instead of every cell; '
                         'much less time, disk and memory when a single anchor is trained; needs --run-augmentation')
    p.add_argument('--run-identification', action='store_true',
                    help='Run identification/tad_detection.py after training to identify single-cell TADs')
    p.add_argument('--run-visualization', action='store_true',
                    help='Run visualization/tad.py after identification: visualize, print validation metrics, and write TAD result CSVs')

    p.add_argument('--logfile', type=str, default=None,
                    help='Log file path, defaults to output_dir/CellTAD.log')

    return p


def validate_args(args):
    if args.augmentation_anchor_only and not args.run_augmentation:
        raise ValueError('--augmentation-anchor-only only applies together with --run-augmentation')
    if args.demo:
        if args.run_augmentation or args.augmentation_only is not None or args.augmentation_start_from is not None:
            raise ValueError('--demo already contains the augmented maps; do not combine it with '
                             '--run-augmentation / --augmentation-only / --augmentation-start-from')
        if args.hic_dir:
            raise ValueError('--demo cannot be combined with --hic-dir')
        if args.anchor_cell is not None and args.anchor_cell not in DEMO_DATASETS:
            raise ValueError(f"--demo has no demo for anchor cell '{args.anchor_cell}'; "
                             f"available: {', '.join(DEMO_DATASETS)}")


def find_chrom_sizes_file(path):
    """A relative path is looked up in the current directory first, then in the project root."""
    candidates = [path] if os.path.isabs(path) else [path, os.path.join(main_dir, path)]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise ValueError(f"chromosome size file not found: {path} (looked in: "
                     f"{', '.join(os.path.abspath(c) for c in candidates)})")


def build_cfg_from_args(args):
    validate_args(args)

    cfg = get_default_config()
    if args.demo:
        if args.anchor_cell:
            cfg['anchor_cell'] = args.anchor_cell
        cfg = apply_demo_mode(cfg)

    overrides = {
        'base_dir': args.base_dir, 'anchor_cell': args.anchor_cell,
        'chrom': args.chrom, 'resolution': args.resolution,
        'output_dir': args.output_dir, 'num_epochs': args.num_epochs,
        'visualization_outdir': args.visualization_outdir,
        'hic_dir': args.hic_dir, 'chrom_sizes_file': args.chrom_sizes,
        'device': args.device,
        'augmentation_force': args.augmentation_force if args.augmentation_force else None,
        'augmentation_start_from': args.augmentation_start_from,
        'augmentation_only': args.augmentation_only,
        'augmentation_anchor_only': True if args.augmentation_anchor_only else None,
        'run_augmentation_pipeline': True if args.run_augmentation else None,
        'run_identification': True if args.run_identification else None,
        'run_visualization': True if args.run_visualization else None,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v

    if args.demo and args.chrom_sizes is None:
        demo_sizes = DEMO_DATASETS[cfg['anchor_cell']].get('chrom_sizes')
        if demo_sizes:
            cfg['chrom_sizes_file'] = demo_sizes

    if args.visualization_outdir is None and args.output_dir:
        cfg['visualization_outdir'] = os.path.join(cfg['output_dir'], 'visualization', 'tad')

    if cfg.get('chrom_sizes_file'):
        cfg['chrom_sizes_file'] = find_chrom_sizes_file(cfg['chrom_sizes_file'])

    cfg = resolve_paths(cfg)
    return cfg


def main():
    args = create_parser().parse_args()
    try:
        cfg = build_cfg_from_args(args)
    except ValueError as e:
        print(f'Error: {e}', file=sys.stderr)
        return 1

    os.makedirs(cfg['output_dir'], exist_ok=True)
    logfile = args.logfile or os.path.join(cfg['output_dir'], 'CellTAD.log')
    logger = Logger(logfile, rank=0, verbose_threshold=1)
    logger.dump_args(args)

    if cfg.get('chrom_sizes_file'):
        os.environ['CELLTAD_CHROM_SIZES_FILE'] = cfg['chrom_sizes_file']

    if cfg.get('run_augmentation_pipeline', False):
        logger.write('Step A: running augmentation pipeline (7 steps)...', verbose_level=1)
        from train.augmentation_loader import run_augmentation_pipeline
        ok = run_augmentation_pipeline(
            base_dir=cfg['base_dir'],
            notebook_cwd=cfg.get('augmentation_notebook_cwd'),
            force=cfg.get('augmentation_force', False),
            start_from=cfg.get('augmentation_start_from', 1),
            only=cfg.get('augmentation_only'),
            hic_dir=cfg.get('hic_dir'), chrom_sizes_file=cfg.get('chrom_sizes_file'),
            chrom=cfg['chrom'], resolution=cfg['resolution'],
            anchor_cell=cfg['anchor_cell'] if cfg.get('augmentation_anchor_only') else None,
        )
        if not ok:
            logger.write('Step A failed, aborting.', verbose_level=1)
            logger.flush()
            return 1
        if cfg.get('hic_dir'):
            try:
                cfg = resolve_paths(cfg)
            except ValueError as e:
                logger.write(f'{e}\n  Cells removed in Step 0 are listed in '
                             f"{os.path.join(cfg['base_dir'], 'hic_to_pairs_low_quality.tsv')}", verbose_level=1)
                logger.flush()
                return 1

    if cfg.get('hic_dir') and not os.path.exists(cfg['anchor_path']):
        logger.write(f"Anchor pairs file not found: {cfg['anchor_path']}\n"
                     f"  Run with --run-augmentation so Step 0 converts the .hic files, or check that the anchor was "
                     f"not excluded (see hic_to_pairs_low_quality.tsv in {cfg['base_dir']}).", verbose_level=1)
        logger.flush()
        return 1

    logger.write('Step B: training CellTAD embedding...', verbose_level=1)
    from train.train_CellTAD import main as train_main
    model, logs, final_embedding = train_main(cfg)

    if cfg.get('run_identification', False) or cfg.get('run_visualization', False):
        logger.write('Step C+D: detecting & visualizing single-cell TADs '
                      '(visualization.tad.main)...', verbose_level=1)
        from visualization.tad import main as run_tad_identification_and_viz
        run_tad_identification_and_viz(
            edir=cfg['output_dir'], cell=cfg['anchor_cell'], chrom=cfg['chrom'],
            res=cfg['resolution'], hic_file=cfg['anchor_path'],
            outdir=cfg['visualization_outdir'], regions=cfg['visualization_regions'],
        )

    logger.write('CellTAD pipeline finished.', verbose_level=1)
    logger.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())