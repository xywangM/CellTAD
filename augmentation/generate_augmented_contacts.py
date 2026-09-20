#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
import pandas as pd
import pickle
import time
from collections import defaultdict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config import get_default_config
from chrom_sizes import load_chrom_sizes

_cfg = get_default_config()

BASE_DIR = os.environ.get("CELLTAD_BASE_DIR", _cfg['base_dir'])
PAIRS_DIR = os.path.join(BASE_DIR, "GSE279583_extracted")

SIMILAR_DIR = os.path.join(BASE_DIR, "similar")

OUTPUT_BASE_DIR = os.path.join(BASE_DIR, "enhanced_maps")

TARGET_CHROM = os.environ.get("CELLTAD_CHROM", _cfg['chrom'])
TAD_FILE = os.path.join(BASE_DIR, "tadgate_results", "tad_boundaries", f"{TARGET_CHROM}_TADs.csv")
TOP_CELLS_FILE = os.path.join(SIMILAR_DIR, "tad_top10_similar_cells.json")

RESOLUTION = int(os.environ.get("CELLTAD_RESOLUTION", _cfg['resolution']))
N_AUGMENTATIONS = int(os.environ.get("CELLTAD_N_AUG", _cfg['n_enhanced_candidates']))
ANCHOR_CELL = os.environ.get("CELLTAD_ANCHOR_CELL") or None

AUGMENTATION_CONFIG = {
    'noise_std': 0.01,
}

CHROM_SIZES = load_chrom_sizes()


def get_cell_number_from_id(cell_id):
    """Extract the numeric cell index from a cell ID."""
    import re
    match = re.search(r'GM-800U_(\d+)', str(cell_id))
    if match:
        return int(match.group(1))
    match = re.search(r'_(\d+)$', str(cell_id))
    if match:
        return int(match.group(1))
    match = re.search(r'cell_(\d+)', str(cell_id))
    if match:
        return int(match.group(1))
    if str(cell_id).isdigit():
        return int(cell_id)
    return None


def resolve_anchor_cell_id(anchor, base_dir, known_ids):
    """Map an anchor name (cell_XXX, GM-800U_XXX or an original .hic name from cell_id_map.tsv) to the cell id used in the kNN results."""
    anchor = str(anchor).strip()
    if anchor in known_ids:
        return anchor
    map_path = os.path.join(base_dir, "cell_id_map.tsv")
    if os.path.exists(map_path):
        with open(map_path) as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 2 and p[0] != "cell_id" and not line.startswith("#") and p[1] == anchor and p[0] in known_ids:
                    return p[0]
    raise ValueError(
        f"anchor cell '{anchor}' is not among the {len(known_ids)} cells of the kNN results; "
        f"cells removed in Step 0 are listed in {os.path.join(base_dir, 'hic_to_pairs_low_quality.tsv')}")


def cell_chrom_dir(output_base_dir, cell_id, chrom):
    cell_num = get_cell_number_from_id(cell_id)
    if cell_num is None:
        return None
    return os.path.join(output_base_dir, f"cell_{cell_num:03d}", chrom)


def check_cell_completed(output_base_dir, cell_id, chrom, n_augmentations):
    """Check whether a cell's augmented maps have already been fully generated."""
    chrom_dir = cell_chrom_dir(output_base_dir, cell_id, chrom)
    if chrom_dir is None or not os.path.exists(chrom_dir):
        return False

    for i in range(n_augmentations):
        filepath = os.path.join(chrom_dir, f"enhanced_top{i+1}.npy")
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return False

    return True


def find_pairs_file(pairs_dir, cell_num):
    """Locate the pairs file for a given cell."""
    patterns = [
        f"GM-800U_{cell_num:03d}.allValidPairs.txt",
        f"cell_{cell_num:03d}.allValidPairs.txt",
        f"{cell_num:03d}.allValidPairs.txt",
    ]
    for pattern in patterns:
        filepath = os.path.join(pairs_dir, pattern)
        if os.path.exists(filepath):
            return filepath
    return None


def get_all_cell_ids():
    """Get the cell ids from the kNN results."""
    pkl_file = os.path.join(SIMILAR_DIR, "similar_cells.pkl")

    if not os.path.exists(pkl_file):
        raise FileNotFoundError(f"kNN result file not found: {pkl_file}")

    with open(pkl_file, 'rb') as f:
        results = pickle.load(f)

    cell_ids = results['cell_ids']
    print(f"Loaded {len(cell_ids)} cell IDs from kNN results")

    return cell_ids


def load_anchor_matrix(pairs_file, chrom, resolution):
    """Load an anchor cell's full Hi-C matrix."""
    chrom_size = CHROM_SIZES.get(chrom)
    if chrom_size is None:
        raise ValueError(f"Unknown chromosome: {chrom}")

    n_bins = (chrom_size // resolution) + 1
    matrix = np.zeros((n_bins, n_bins), dtype=np.float32)

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


def apply_global_gaussian_noise(matrix, noise_std, seed=None):
    """Add symmetric Gaussian noise (std=noise_std) and clip values to >= 0."""
    if seed is not None:
        np.random.seed(seed)

    noise = (np.random.randn(*matrix.shape) * noise_std).astype(matrix.dtype)

    noise_upper = np.triu(noise)
    noise_symmetric = noise_upper + noise_upper.T - np.diag(np.diag(noise_upper))

    matrix_noisy = matrix + noise_symmetric
    matrix_noisy = np.maximum(matrix_noisy, 0)

    return matrix_noisy


def generate_enhanced_maps_for_cell(anchor_matrix, tad_df, top_cells_map, pairs_dir,
                                    chrom, resolution, n_augmentations, aug_config):
    """Generate n_augmentations noisy views of the anchor matrix with TAD blocks replaced by candidate cells' blocks."""
    mat_size = anchor_matrix.shape[0]

    replacement_stats = {
        'total_tads': 0,
        'successful_replacements': [0] * n_augmentations,
        'failed_tads': [[] for _ in range(n_augmentations)],
        'replaced_regions': [[] for _ in range(n_augmentations)],
    }

    chrom_tads = tad_df[tad_df['chromosome'] == chrom].copy()

    if len(chrom_tads) == 0:
        return [anchor_matrix.copy() for _ in range(n_augmentations)], replacement_stats

    replacement_stats['total_tads'] = len(chrom_tads)

    print(f"    Collecting the TAD blocks needed from each similar cell (full top-10 pool per TAD)...")
    blocks_needed = defaultdict(set)
    for _, row in chrom_tads.iterrows():
        tad_key = f"{chrom}_TAD{row['TAD_id']}"
        if tad_key not in top_cells_map:
            continue
        bin_start = int(row['start_pos'] // resolution)
        bin_end = int((row['end_pos'] + resolution - 1) // resolution)
        if bin_end - bin_start <= 0 or bin_end > mat_size:
            continue
        for cell_num in top_cells_map[tad_key]:
            blocks_needed[cell_num].add((bin_start, bin_end))

    print(f"    Need blocks from {len(blocks_needed)} similar cells...")

    similar_blocks = {}
    loaded_count = 0
    for cell_num, blocks in blocks_needed.items():
        pairs_file = find_pairs_file(pairs_dir, cell_num)
        if pairs_file is None:
            continue

        try:
            matrix, _ = load_anchor_matrix(pairs_file, chrom, resolution)
            cell_blocks = {}
            for bin_start, bin_end in blocks:
                if bin_end <= matrix.shape[0]:
                    cell_blocks[(bin_start, bin_end)] = matrix[bin_start:bin_end, bin_start:bin_end].copy()
            similar_blocks[cell_num] = cell_blocks
            del matrix
            loaded_count += 1

            if loaded_count % 10 == 0:
                print(f"      Loaded {loaded_count}/{len(blocks_needed)} cells...")
        except Exception as e:
            print(f"      Failed to load cell {cell_num}: {e}")
            continue

    print(f"    Loaded blocks from {len(similar_blocks)}/{len(blocks_needed)} cells")

    noise_std = aug_config.get('noise_std', 0.01)
    enhanced_mats = []

    for aug_idx in range(n_augmentations):
        view_matrix = apply_global_gaussian_noise(
            anchor_matrix, noise_std, seed=aug_idx * 10000
        )

        rng = np.random.RandomState(aug_idx * 10000 + 1)

        for _, row in chrom_tads.iterrows():
            tad_id = row['TAD_id']
            start = row['start_pos']
            end = row['end_pos']

            bin_start = start // resolution
            bin_end = (end + resolution - 1) // resolution
            bin_num = bin_end - bin_start

            if bin_num <= 0 or bin_end > mat_size:
                replacement_stats['failed_tads'][aug_idx].append(tad_id)
                continue

            tad_key = f"{chrom}_TAD{tad_id}"

            if tad_key not in top_cells_map or len(top_cells_map[tad_key]) == 0:
                replacement_stats['failed_tads'][aug_idx].append(tad_id)
                continue

            candidate_pool = top_cells_map[tad_key]

            cell_num = candidate_pool[rng.randint(len(candidate_pool))]

            submat = similar_blocks.get(cell_num, {}).get((int(bin_start), int(bin_end)))

            if submat is None or submat.shape != (bin_num, bin_num):
                replacement_stats['failed_tads'][aug_idx].append(tad_id)
                continue

            view_matrix[bin_start:bin_end, bin_start:bin_end] = submat

            replacement_stats['replaced_regions'][aug_idx].append({
                'tad_id': tad_id,
                'bin_start': int(bin_start),
                'bin_end': int(bin_end),
                'cell_num': int(cell_num)
            })
            replacement_stats['successful_replacements'][aug_idx] += 1

        enhanced_mats.append(view_matrix)
        print(f"    View {aug_idx+1}/{n_augmentations}: "
              f"replaced {replacement_stats['successful_replacements'][aug_idx]}/{len(chrom_tads)} TADs")

    return enhanced_mats, replacement_stats


def save_enhanced_maps_hierarchical(enhanced_mats, output_base_dir, cell_id, chrom):
    """Save augmented maps in the hierarchical directory layout."""
    cell_num = get_cell_number_from_id(cell_id)
    if cell_num is None:
        raise ValueError(f"Could not parse cell ID: {cell_id}")

    chrom_dir = cell_chrom_dir(output_base_dir, cell_id, chrom)
    os.makedirs(chrom_dir, exist_ok=True)

    saved_files = []

    for i, mat in enumerate(enhanced_mats):
        filename = f"enhanced_top{i+1}.npy"
        filepath = os.path.join(chrom_dir, filename)

        np.save(filepath, mat.astype(np.float32, copy=False))
        saved_files.append(filepath)

    return saved_files, chrom_dir


def process_all_cells(cell_ids, tad_df, top_cells_map, pairs_dir,
                     output_base_dir, chrom, resolution, n_augmentations,
                     augmentation_config):
    """Batch-process all cells to generate augmented maps."""
    start_time = time.time()

    overall_stats = {
        'total_cells': len(cell_ids),
        'successful_cells': 0,
        'skipped_cells': 0,
        'failed_cells': [],
        'cell_details': {}
    }

    for idx, cell_id in enumerate(cell_ids, 1):
        print(f"\n{'='*80}")
        print(f"Processing cell [{idx}/{len(cell_ids)}]: {cell_id}")
        print(f"{'='*80}")

        try:
            if check_cell_completed(output_base_dir, cell_id, chrom, n_augmentations):
                print(f"  Already complete, skipping (enhanced_maps/{cell_id}/{chrom}/)")
                skip_stats = os.path.join(cell_chrom_dir(output_base_dir, cell_id, chrom), "stats.json")
                if not os.path.exists(skip_stats):
                    with open(skip_stats, 'w') as f:
                        json.dump({'note': 'maps already present; replacement statistics not available'}, f)
                overall_stats['skipped_cells'] += 1
                overall_stats['successful_cells'] += 1
                continue

            cell_start = time.time()

            cell_num = get_cell_number_from_id(cell_id)
            if cell_num is None:
                raise ValueError(f"Could not parse cell number from cell_id: {cell_id}")

            pairs_file = find_pairs_file(pairs_dir, cell_num)
            if pairs_file is None:
                raise FileNotFoundError(f"Pairs file not found for {cell_id} (number {cell_num})")

            print(f"  Loading anchor cell's {chrom} contact matrix...")
            load_start = time.time()
            anchor_matrix, contact_count = load_anchor_matrix(pairs_file, chrom, resolution)
            load_time = time.time() - load_start
            print(f"  Matrix loaded: {anchor_matrix.shape}, "
                  f"total contacts={contact_count} (time: {load_time:.1f}s)")

            print(f"  Generating {n_augmentations} augmented maps...")
            aug_start = time.time()
            enhanced_mats, stats = generate_enhanced_maps_for_cell(
                anchor_matrix, tad_df, top_cells_map, pairs_dir,
                chrom, resolution, n_augmentations, augmentation_config
            )
            aug_time = time.time() - aug_start
            print(f"  Augmented maps generated (time: {aug_time:.1f}s)")

            print(f"  Saving...")
            save_start = time.time()
            saved_files, save_dir = save_enhanced_maps_hierarchical(
                enhanced_mats, output_base_dir, cell_id, chrom
            )
            save_time = time.time() - save_start
            print(f"  Saved: {len(saved_files)} files (time: {save_time:.1f}s)")
            print(f"    Directory: {save_dir}")

            cell_total_time = time.time() - cell_start
            print(f"  Cell total time: {cell_total_time:.1f}s ({cell_total_time/60:.2f} min)")

            overall_stats['successful_cells'] += 1
            overall_stats['cell_details'][cell_id] = {
                'cell_num': cell_num,
                'contact_count': contact_count,
                'tad_stats': stats,
                'output_dir': save_dir,
                'n_files': len(saved_files),
                'processing_time': cell_total_time
            }

            cell_stats_file = os.path.join(save_dir, "stats.json")
            with open(cell_stats_file, 'w') as f:
                json.dump(stats, f, indent=2)

        except Exception as e:
            print(f"  Failed: {str(e)}")
            import traceback
            traceback.print_exc()
            overall_stats['failed_cells'].append({
                'cell_id': cell_id,
                'reason': str(e)
            })
            continue

    total_elapsed = time.time() - start_time

    return overall_stats, total_elapsed


def main():
    print("=" * 80)
    print("Batch-generate augmented Hi-C maps - paper Methods-aligned, resumable")
    print("=" * 80)
    print(f"Target chromosome: {TARGET_CHROM}")
    print(f"Resolution: {RESOLUTION} bp")
    print(f"Augmented maps per cell: {N_AUGMENTATIONS}")
    print(f"Cells: {'anchor only (' + ANCHOR_CELL + ')' if ANCHOR_CELL else 'all cells in the kNN results'}")
    print(f"Output layout: enhanced_maps/cell_XXX/{TARGET_CHROM}/enhanced_topN.npy")
    print("\nAugmentation strategy:")
    print(f"  1. Global Gaussian noise: C~(t) = C_c + eps(t), eps_ij~N(0,sigma^2), sigma={AUGMENTATION_CONFIG['noise_std']}")
    print(f"  2. Uniformly sample one candidate cell from each TAD's top-10 pool for replacement")
    print("=" * 80)

    try:
        print("\nStep 1: Get cell list")
        print("-" * 80)
        cell_ids = get_all_cell_ids()
        if ANCHOR_CELL:
            anchor_id = resolve_anchor_cell_id(ANCHOR_CELL, BASE_DIR, known_ids=cell_ids)
            print(f"  Anchor-only mode: {ANCHOR_CELL} -> {anchor_id}")
            cell_ids = [anchor_id]
        print(f"  Cell count: {len(cell_ids)}")
        print(f"  Example cells: {cell_ids[:5]}")

        print("\nStep 2: Load TAD boundaries")
        print("-" * 80)

        if not os.path.exists(TAD_FILE):
            print(f"Error: TAD file not found: {TAD_FILE}")
            return 1

        tad_df = pd.read_csv(TAD_FILE)
        print(f"Loaded TAD file: {os.path.basename(TAD_FILE)}")
        print(f"  Total TADs: {len(tad_df)}")
        print(f"  Columns: {list(tad_df.columns)}")

        print("\nStep 3: Load TAD-to-similar-cells mapping (top-10 candidate pool)")
        print("-" * 80)

        if not os.path.exists(TOP_CELLS_FILE):
            print(f"Error: similar cells file not found: {TOP_CELLS_FILE}")
            return 1

        with open(TOP_CELLS_FILE, 'r') as f:
            top_cells_map = json.load(f)

        print(f"Loaded file: {os.path.basename(TOP_CELLS_FILE)}")
        print(f"  TAD count: {len(top_cells_map)}")

        sample_keys = list(top_cells_map.keys())[:3]
        print(f"  Example TADs:")
        for key in sample_keys:
            print(f"    {key}: {top_cells_map[key][:3]}...")

        print("\nStep 4: Batch process all cells (resumable)")
        print("-" * 80)

        overall_stats, total_time = process_all_cells(
            cell_ids, tad_df, top_cells_map, PAIRS_DIR,
            OUTPUT_BASE_DIR, TARGET_CHROM, RESOLUTION, N_AUGMENTATIONS,
            AUGMENTATION_CONFIG
        )

        print("\n" + "=" * 80)
        print("Saving overall statistics")
        print("=" * 80)

        if ANCHOR_CELL:
            print("Anchor-only mode: batch_processing_summary.json is not written "
                  "(per-cell stats.json marks the anchor as done)")
        else:
            summary_file = os.path.join(OUTPUT_BASE_DIR, "batch_processing_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(overall_stats, f, indent=2)
            print(f"Statistics saved: {summary_file}")

        print("\n" + "=" * 80)
        print("Batch processing done")
        print("=" * 80)
        print(f"\nProcessing summary:")
        print(f"  Total cells: {overall_stats['total_cells']}")
        print(f"  Skipped (already complete): {overall_stats.get('skipped_cells', 0)}")
        print(f"  Succeeded: {overall_stats['successful_cells']}")
        print(f"  Failed: {len(overall_stats['failed_cells'])}")

        if total_time > 0:
            print(f"  Time this run: {total_time/60:.1f} min")
            new_processed = overall_stats['successful_cells'] - overall_stats.get('skipped_cells', 0)
            if new_processed > 0:
                print(f"  Average speed: {new_processed/total_time*60:.1f} cells/min")

        if overall_stats['failed_cells']:
            print(f"\nFailed cells:")
            for failed in overall_stats['failed_cells'][:10]:
                print(f"  - {failed['cell_id']}: {failed['reason']}")
            if len(overall_stats['failed_cells']) > 10:
                print(f"  ... and {len(overall_stats['failed_cells'])-10} more")

        print(f"\nOutput dir: {OUTPUT_BASE_DIR}")
        print(f"Directory layout example:")
        print(f"  enhanced_maps/")
        print(f"    |-- cell_001/")
        print(f"    |   `-- chr1/")
        print(f"    |       |-- enhanced_top1.npy")
        print(f"    |       |-- enhanced_top2.npy")
        print(f"    |       |-- ...")
        print(f"    |       `-- stats.json (replacement stats for all augmented maps of this cell)")
        print(f"    |-- cell_002/")
        print(f"    |   `-- chr1/...")
        print(f"    `-- batch_processing_summary.json")

        print("=" * 80)

        if ANCHOR_CELL and overall_stats['failed_cells']:
            return 1
        return 0

    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())