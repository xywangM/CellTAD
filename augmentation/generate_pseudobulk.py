#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import time
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config import get_default_config
from chrom_sizes import load_chrom_sizes

_cfg = get_default_config()

BASE_DIR = os.environ.get("CELLTAD_BASE_DIR", _cfg['base_dir'])
PAIRS_DIR = os.path.join(BASE_DIR, "GSE279583_extracted")
BULK_OUTPUT_DIR = os.path.join(BASE_DIR, "tadgate_results", "bulk_hic")

HIC_DIR = os.environ.get("CELLTAD_HIC_DIR", _cfg.get('hic_dir'))

TARGET_CHROM = os.environ.get("CELLTAD_CHROM", _cfg['chrom'])
RESOLUTION = int(os.environ.get("CELLTAD_RESOLUTION", _cfg['resolution']))

CHROM_SIZES = load_chrom_sizes()


def find_all_pairs_files(pairs_dir):
    """Find every single-cell .allValidPairs.txt file under PAIRS_DIR."""
    pattern = os.path.join(pairs_dir, "*.allValidPairs.txt")
    return sorted(glob.glob(pattern))


def load_cell_matrix(pairs_file, chrom, resolution, n_bins):
    """Build a single cell's contact matrix for the given chromosome from its pairs file."""
    matrix = np.zeros((n_bins, n_bins), dtype=np.float64)
    contact_count = 0

    with open(pairs_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue

            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue

            try:
                chr1 = parts[1]
                pos1 = int(parts[2])
                chr2 = parts[3]
                pos2 = int(parts[4])

                if chr1 != chrom or chr2 != chrom:
                    continue

                i = pos1 // resolution
                j = pos2 // resolution

                if i < n_bins and j < n_bins:
                    matrix[i, j] += 1
                    if i != j:
                        matrix[j, i] += 1
                    contact_count += 1

            except (ValueError, IndexError):
                continue

    return matrix, contact_count


def aggregate_bulk_hic(pairs_dir, chrom, resolution):
    """Aggregate all single-cell Hi-C contact matrices into a pseudo-bulk matrix: C_bulk = sum_i C_i."""
    chrom_size = CHROM_SIZES.get(chrom)
    if chrom_size is None:
        raise ValueError(f"Unknown chromosome: {chrom}")

    n_bins = (chrom_size // resolution) + 1

    pairs_files = find_all_pairs_files(pairs_dir)
    print(f"  Found {len(pairs_files)} single-cell pairs files")

    if len(pairs_files) == 0:
        raise FileNotFoundError(f"No *.allValidPairs.txt files found under {pairs_dir}")

    bulk_matrix = np.zeros((n_bins, n_bins), dtype=np.float64)
    n_cells_used = 0

    start_time = time.time()

    for idx, pairs_file in enumerate(pairs_files, 1):
        cell_matrix, contact_count = load_cell_matrix(
            pairs_file, chrom, resolution, n_bins
        )

        if contact_count == 0:
            continue

        bulk_matrix += cell_matrix
        n_cells_used += 1

        if idx % 10 == 0 or idx == len(pairs_files):
            elapsed = time.time() - start_time
            print(f"    Progress: {idx}/{len(pairs_files)} "
                  f"(aggregated {n_cells_used} cells, elapsed {elapsed/60:.1f} min)")

    print(f"\n  Aggregation done")
    print(f"    Cells used: {n_cells_used}/{len(pairs_files)}")
    print(f"    Bulk matrix shape: {bulk_matrix.shape}")
    print(f"    Total bulk contacts (deduplicated): {int(bulk_matrix.sum() // 2)}")

    return bulk_matrix, n_cells_used


def main():
    print("=" * 80)
    print("Module 1 - Step 1: aggregate all cells' Hi-C maps into pseudo-bulk Hi-C")
    print("=" * 80)
    print(f"Input dir: {PAIRS_DIR}")
    print(f"Chromosome: {TARGET_CHROM}")
    print(f"Resolution: {RESOLUTION} bp")
    print("=" * 80)

    if HIC_DIR:
        if _THIS_DIR not in sys.path:
            sys.path.insert(0, _THIS_DIR)
        from convert_hic_to_pairs import run as convert_hic_to_pairs
        if convert_hic_to_pairs(HIC_DIR, PAIRS_DIR) != 0:
            return 1

    os.makedirs(BULK_OUTPUT_DIR, exist_ok=True)

    bulk_matrix, n_cells_used = aggregate_bulk_hic(
        PAIRS_DIR, TARGET_CHROM, RESOLUTION
    )

    output_path = os.path.join(
        BULK_OUTPUT_DIR, f"{TARGET_CHROM}_bulk_{RESOLUTION}bp.npy"
    )
    np.save(output_path, bulk_matrix)

    print(f"\n  Pseudo-bulk matrix saved: {output_path}")
    print(f"    Cells used: {n_cells_used}")

    return 0


if __name__ == '__main__':
    exit(main())