#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import numpy as np
import cooler
import pandas as pd
import os
import sys
import json
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import get_default_config
from chrom_sizes import load_chrom_sizes

_cfg = get_default_config()

BASE_DIR = os.environ.get("CELLTAD_BASE_DIR", _cfg['base_dir'])
ENHANCED_MAPS_DIR = os.path.join(BASE_DIR, "enhanced_maps")

RESOLUTION = int(os.environ.get("CELLTAD_RESOLUTION", _cfg['resolution']))
TARGET_CHROM = os.environ.get("CELLTAD_CHROM", _cfg['chrom'])
N_AUGMENTATIONS = int(os.environ.get("CELLTAD_N_AUG", _cfg['n_enhanced_candidates']))
ANCHOR_CELL = os.environ.get("CELLTAD_ANCHOR_CELL") or None

CHROM_SIZES = load_chrom_sizes()


def resolve_anchor_dir(anchor, base_dir):
    """Map an anchor name (cell_XXX, GM-800U_XXX or an original .hic name from cell_id_map.tsv) to its enhanced_maps/ directory name."""
    anchor = str(anchor).strip()
    cell_id = anchor
    map_path = os.path.join(base_dir, "cell_id_map.tsv")
    if os.path.exists(map_path):
        with open(map_path) as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 2 and p[0] != "cell_id" and not line.startswith("#") and p[1] == anchor:
                    cell_id = p[0]
                    break
    m = re.fullmatch(r"(?:cell_|GM-800U_)?(\d+)", cell_id)
    if m is None:
        raise ValueError(
            f"anchor cell '{anchor}' is not in {map_path}; "
            f"use cell_XXX or an original .hic name listed in that file")
    return f"cell_{int(m.group(1)):03d}"


def create_bins_dataframe(n_bins, chrom, resolution, chrom_size):
    bin_starts = [i * resolution for i in range(n_bins)]
    bin_ends = [(i + 1) * resolution for i in range(n_bins)]
    if bin_ends[-1] > chrom_size:
        bin_ends[-1] = chrom_size
    return pd.DataFrame({"chrom": [chrom] * n_bins, "start": bin_starts, "end": bin_ends})


def extract_pixels_from_matrix(matrix):
    n_bins = matrix.shape[0]
    rows, cols = np.triu_indices(n_bins)
    counts = matrix[rows, cols]
    nonzero_mask = counts > 0
    rows = rows[nonzero_mask]; cols = cols[nonzero_mask]; counts = counts[nonzero_mask]
    pixels_df = pd.DataFrame({
        "bin1_id": rows.astype(np.int32),
        "bin2_id": cols.astype(np.int32),
        "count": counts.astype(np.float32)
    })
    return pixels_df.sort_values(['bin1_id', 'bin2_id']).reset_index(drop=True)


def convert_npy_to_cool(npy_path, cool_path, chrom, resolution, chrom_size):
    try:
        matrix = np.load(npy_path)
        n_bins = matrix.shape[0]
        bins_df = create_bins_dataframe(n_bins, chrom, resolution, chrom_size)
        pixels_df = extract_pixels_from_matrix(matrix)
        total_contacts = len(pixels_df)

        os.makedirs(os.path.dirname(cool_path), exist_ok=True)
        if os.path.exists(cool_path):
            os.remove(cool_path)

        cooler.create_cooler(
            cool_path, bins=bins_df, pixels=pixels_df,
            dtypes={'count': np.float32}, ordered=True
        )

        stats = {
            'n_bins': n_bins, 'total_contacts': total_contacts,
            'density': total_contacts / (n_bins * (n_bins + 1) / 2) * 100,
            'matrix_sum': float(matrix.sum()), 'matrix_mean': float(matrix.mean()),
            'matrix_std': float(matrix.std())
        }
        return True, stats
    except Exception as e:
        print(f"      Conversion failed: {str(e)}")
        return False, None


def get_all_cell_dirs(enhanced_maps_dir, only_dir=None):
    cell_dirs = []
    if not os.path.exists(enhanced_maps_dir):
        raise FileNotFoundError(f"Enhanced maps directory not found: {enhanced_maps_dir}")
    for item in sorted(os.listdir(enhanced_maps_dir)):
        if only_dir is not None and item != only_dir:
            continue
        item_path = os.path.join(enhanced_maps_dir, item)
        if os.path.isdir(item_path) and item.startswith('cell_'):
            try:
                cell_num = int(item.split('_')[1])
                cell_dirs.append((item_path, cell_num))
            except (ValueError, IndexError):
                continue
    return cell_dirs


def process_single_cell(cell_dir, cell_num, chrom, resolution, chrom_size, n_augmentations):
    chrom_dir = os.path.join(cell_dir, chrom)
    if not os.path.exists(chrom_dir):
        return {'cell_num': cell_num, 'status': 'failed', 'reason': f'{chrom} directory not found',
                'successful_conversions': 0, 'failed_conversions': 0, 'files': []}

    cell_stats = {'cell_num': cell_num, 'status': 'processing',
                  'successful_conversions': 0, 'failed_conversions': 0, 'files': []}

    for i in range(1, n_augmentations + 1):
        npy_filename = f"enhanced_top{i}.npy"
        npy_path = os.path.join(chrom_dir, npy_filename)
        cool_filename = f"enhanced_top{i}.cool"
        cool_path = os.path.join(chrom_dir, cool_filename)

        if not os.path.exists(npy_path):
            cell_stats['failed_conversions'] += 1
            cell_stats['files'].append({'index': i, 'npy_file': npy_filename,
                                        'cool_file': cool_filename, 'success': False,
                                        'reason': 'npy file not found'})
            continue

        success, stats = convert_npy_to_cool(npy_path, cool_path, chrom, resolution, chrom_size)
        if success:
            cell_stats['successful_conversions'] += 1
            cell_stats['files'].append({'index': i, 'npy_file': npy_filename,
                                        'cool_file': cool_filename, 'success': True, 'stats': stats})
        else:
            cell_stats['failed_conversions'] += 1
            cell_stats['files'].append({'index': i, 'npy_file': npy_filename,
                                        'cool_file': cool_filename, 'success': False,
                                        'reason': 'conversion error'})

    if cell_stats['successful_conversions'] == n_augmentations:
        cell_stats['status'] = 'success'
    elif cell_stats['successful_conversions'] > 0:
        cell_stats['status'] = 'partial'
    else:
        cell_stats['status'] = 'failed'
    return cell_stats


def batch_convert_all_cells(enhanced_maps_dir, chrom, resolution, chrom_size, n_augmentations, only_dir=None):
    print("\n" + "=" * 80)
    print("Batch convert enhanced maps to .cool")
    print("=" * 80)
    cell_dirs = get_all_cell_dirs(enhanced_maps_dir, only_dir)
    if len(cell_dirs) == 0:
        print(f"  No cell directories found{' for ' + only_dir if only_dir else ''}")
        return {}, 0.0

    print(f"  Found {len(cell_dirs)} cell directories")
    overall_stats = {'total_cells': len(cell_dirs), 'successful_cells': 0, 'partial_cells': 0,
                     'failed_cells': 0, 'total_conversions': 0, 'successful_conversions': 0,
                     'failed_conversions': 0, 'cell_details': {}}
    start_time = time.time()

    for idx, (cell_dir, cell_num) in enumerate(cell_dirs):
        cell_name = os.path.basename(cell_dir)
        cell_stats = process_single_cell(cell_dir, cell_num, chrom, resolution, chrom_size, n_augmentations)
        overall_stats['cell_details'][cell_name] = cell_stats

        if cell_stats['status'] == 'success':
            overall_stats['successful_cells'] += 1
        elif cell_stats['status'] == 'partial':
            overall_stats['partial_cells'] += 1
        else:
            overall_stats['failed_cells'] += 1

        overall_stats['total_conversions'] += (cell_stats['successful_conversions'] + cell_stats['failed_conversions'])
        overall_stats['successful_conversions'] += cell_stats['successful_conversions']
        overall_stats['failed_conversions'] += cell_stats['failed_conversions']

        print(f"[{idx+1}/{len(cell_dirs)}] {cell_name}: {cell_stats['status']} "
              f"({cell_stats['successful_conversions']}/{n_augmentations})")

    return overall_stats, time.time() - start_time


def save_batch_report(overall_stats, total_time, output_dir, n_augmentations):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "cool_conversion_report.json")
    with open(json_path, 'w') as f:
        json.dump(overall_stats, f, indent=2)
    print(f"  Report saved (JSON): {json_path}")


def main():
    print("=" * 80)
    print("Batch convert all cells' enhanced maps to .cool")
    print(f"  BASE_DIR={BASE_DIR}  RESOLUTION={RESOLUTION}  "
          f"TARGET_CHROM={TARGET_CHROM}  N_AUGMENTATIONS={N_AUGMENTATIONS}")
    print(f"  Cells: {'anchor only (' + ANCHOR_CELL + ')' if ANCHOR_CELL else 'all cell directories'}")
    print("=" * 80)

    if not os.path.exists(ENHANCED_MAPS_DIR):
        print(f"\nError: enhanced maps directory not found: {ENHANCED_MAPS_DIR}")
        return 1

    chrom_size = CHROM_SIZES.get(TARGET_CHROM)
    if chrom_size is None:
        print(f"\nError: unknown chromosome: {TARGET_CHROM}")
        return 1

    only_dir = None
    if ANCHOR_CELL:
        only_dir = resolve_anchor_dir(ANCHOR_CELL, BASE_DIR)
        print(f"  Anchor-only mode: {ANCHOR_CELL} -> enhanced_maps/{only_dir}/")

    overall_stats, total_time = batch_convert_all_cells(
        ENHANCED_MAPS_DIR, TARGET_CHROM, RESOLUTION, chrom_size, N_AUGMENTATIONS, only_dir
    )

    if only_dir:
        if not overall_stats or overall_stats['successful_cells'] != 1:
            print(f"\nError: conversion for {only_dir} did not complete "
                  f"(run Step 6 for this anchor first, or check the messages above)")
            return 1
        report_dir = os.path.join(ENHANCED_MAPS_DIR, only_dir, TARGET_CHROM)
        save_batch_report(overall_stats, total_time, report_dir, N_AUGMENTATIONS)
    else:
        save_batch_report(overall_stats, total_time, ENHANCED_MAPS_DIR, N_AUGMENTATIONS)

    print(f"\nDone. Succeeded {overall_stats['successful_conversions']}/"
          f"{overall_stats['total_conversions']}")
    return 0


if __name__ == '__main__':
    exit(main())