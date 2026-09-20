#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import json
from scipy.stats import pearsonr
import pickle
from collections import defaultdict
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config import get_default_config

_cfg = get_default_config()

BASE_DIR = os.environ.get("CELLTAD_BASE_DIR", _cfg['base_dir'])
TARGET_CHROM = os.environ.get("CELLTAD_CHROM", _cfg['chrom'])
PAIRS_DIR = os.path.join(BASE_DIR, "GSE279583_extracted")

SIMILAR_BASE_DIR = os.path.join(BASE_DIR, "similar")

TAD_FILE = os.path.join(BASE_DIR, "tadgate_results", "tad_boundaries", f"{TARGET_CHROM}_TADs.csv")
OUTPUT_DIR = SIMILAR_BASE_DIR

RESOLUTION = int(os.environ.get("CELLTAD_RESOLUTION", _cfg['resolution']))
MAX_BIN_SIZE = 500
MIN_BIN_SIZE = 2
TOP_K = 10
N_NEIGHBORS = 10


def load_knn_results(similar_dir):
    """Load kNN indices and cell ids."""
    if not os.path.exists(similar_dir):
        raise FileNotFoundError(f"kNN result directory not found: {similar_dir}")

    indices_file = os.path.join(similar_dir, "neighbor_indices.npy")
    if not os.path.exists(indices_file):
        raise FileNotFoundError(f"kNN result file not found: {indices_file}")

    neighbor_indices = np.load(indices_file)

    pkl_file = os.path.join(similar_dir, "similar_cells.pkl")
    with open(pkl_file, 'rb') as f:
        results = pickle.load(f)
    cell_ids = results['cell_ids']

    print(f"  Loaded kNN results for {len(cell_ids)} cells")
    print(f"    kNN matrix shape: {neighbor_indices.shape}")

    return neighbor_indices, cell_ids


def get_cell_number_from_id(cell_id):
    """Extract the numeric cell index from a cell ID."""
    import re

    match = re.search(r'GM-800U_(\d+)', cell_id)
    if match:
        return int(match.group(1))

    match = re.search(r'_(\d+)$', cell_id)
    if match:
        return int(match.group(1))

    match = re.search(r'cell_(\d+)', cell_id)
    if match:
        return int(match.group(1))

    if str(cell_id).isdigit():
        return int(cell_id)

    return None


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


def extract_tad_submatrix(pairs_file, chrom, bin_start, bin_end, resolution):
    """Deprecated."""
    mat_size = bin_end - bin_start
    matrix = np.zeros((mat_size, mat_size), dtype=np.float64)

    try:
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

                    if bin_start <= i < bin_end and bin_start <= j < bin_end:
                        x = i - bin_start
                        y = j - bin_start
                        matrix[x, y] += 1
                        if x != y:
                            matrix[y, x] += 1

                except (ValueError, IndexError):
                    continue

        return matrix

    except Exception as e:
        print(f"      Failed to read pairs file: {e}")
        return None


def load_cell_chr1_contacts(pairs_file, chrom, resolution):
    """Load all contacts on the target chromosome as {(bin_i, bin_j): count}."""
    contacts = defaultdict(int)

    try:
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

                    bin_i = pos1 // resolution
                    bin_j = pos2 // resolution

                    if bin_i <= bin_j:
                        contacts[(bin_i, bin_j)] += 1
                    else:
                        contacts[(bin_j, bin_i)] += 1

                except (ValueError, IndexError):
                    continue

        return dict(contacts)

    except Exception as e:
        print(f"      Failed to read pairs file: {e}")
        return None


def preload_cell_chr1_contacts(cell_ids, pairs_dir, chrom, resolution):
    """Preload every cell's contacts on the target chromosome as {cell_num: {(bin_i, bin_j): count}}."""
    print(f"\n  Preloading contacts on {chrom} for all cells...")
    print(f"    Cell count: {len(cell_ids)}")

    cell_contacts = {}
    successful_cells = 0

    start_time = time.time()

    for idx, cell_id in enumerate(cell_ids):
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            speed = (idx + 1) / elapsed
            eta = (len(cell_ids) - idx - 1) / speed if speed > 0 else 0
            print(f"    Progress: {idx+1}/{len(cell_ids)} ({100*(idx+1)/len(cell_ids):.1f}%) "
                  f"- speed: {speed:.1f} cells/s - ETA: {eta/60:.1f} min")

        cell_num = get_cell_number_from_id(cell_id)
        if cell_num is None:
            continue

        pairs_file = find_pairs_file(pairs_dir, cell_num)
        if pairs_file is None:
            continue

        contacts = load_cell_chr1_contacts(pairs_file, chrom, resolution)
        if contacts is not None and len(contacts) > 0:
            cell_contacts[cell_num] = contacts
            successful_cells += 1

    elapsed = time.time() - start_time
    print(f"  Preload done. Time: {elapsed/60:.1f} min")
    print(f"    Loaded: {successful_cells}/{len(cell_ids)} cells")

    return cell_contacts


def extract_tad_matrix_from_contacts(contacts, bin_start, bin_end):
    """Extract the TAD-region submatrix from a contacts dict."""
    mat_size = bin_end - bin_start
    matrix = np.zeros((mat_size, mat_size), dtype=np.float64)

    for (bin_i, bin_j), count in contacts.items():
        if bin_start <= bin_i < bin_end and bin_start <= bin_j < bin_end:
            x = bin_i - bin_start
            y = bin_j - bin_start
            matrix[x, y] = count
            if x != y:
                matrix[y, x] = count

    return matrix


def calculate_tad_similarity_optimized(tad_df, neighbor_indices, cell_ids,
                                       cell_contacts, chrom, resolution,
                                       max_bin_size, min_bin_size,
                                       top_k, n_neighbors):
    """Return the top_k most correlated candidate cells for each TAD."""
    tad_results = {}
    processed_count = 0
    skipped_count = 0

    chrom_tads = tad_df[tad_df['chromosome'] == chrom].copy()

    if len(chrom_tads) == 0:
        print(f"  No TADs on {chrom}")
        return tad_results

    print(f"\n  Computing TAD similarity...")
    print(f"  TAD count on {chrom}: {len(chrom_tads)}")

    start_time = time.time()

    for tad_idx, (_, row) in enumerate(chrom_tads.iterrows()):
        if (tad_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            speed = (tad_idx + 1) / elapsed
            eta = (len(chrom_tads) - tad_idx - 1) / speed if speed > 0 else 0
            print(f"    TAD progress: {tad_idx+1}/{len(chrom_tads)} ({100*(tad_idx+1)/len(chrom_tads):.1f}%) "
                  f"- processed: {processed_count} - skipped: {skipped_count} "
                  f"- speed: {speed:.2f} TADs/s - ETA: {eta/60:.1f} min")

        start = row['start_pos']
        end = row['end_pos']
        tad_id = row['TAD_id']

        bin_start = start // resolution
        bin_end = (end + resolution - 1) // resolution
        bin_num = bin_end - bin_start

        if bin_num < min_bin_size or bin_num > max_bin_size:
            skipped_count += 1
            continue

        tad_key = f"{chrom}_TAD{tad_id}"

        all_cell_similarities = []

        for ref_idx, ref_cell_id in enumerate(cell_ids):
            ref_cell_num = get_cell_number_from_id(ref_cell_id)
            if ref_cell_num is None or ref_cell_num not in cell_contacts:
                continue

            ref_mat = extract_tad_matrix_from_contacts(
                cell_contacts[ref_cell_num], bin_start, bin_end
            )

            if np.all(ref_mat == 0):
                continue

            ref_vec = ref_mat.flatten()
            if np.std(ref_vec) == 0:
                continue

            neighbor_idxs = neighbor_indices[ref_idx, :n_neighbors]

            for neighbor_idx in neighbor_idxs:
                neighbor_cell_id = cell_ids[neighbor_idx]
                neighbor_cell_num = get_cell_number_from_id(neighbor_cell_id)

                if neighbor_cell_num is None or neighbor_cell_num not in cell_contacts:
                    continue

                neighbor_mat = extract_tad_matrix_from_contacts(
                    cell_contacts[neighbor_cell_num], bin_start, bin_end
                )

                if np.all(neighbor_mat == 0):
                    continue

                neighbor_vec = neighbor_mat.flatten()
                if np.std(neighbor_vec) == 0:
                    continue

                try:
                    pearson, _ = pearsonr(ref_vec, neighbor_vec)
                    if not np.isnan(pearson):
                        all_cell_similarities.append((neighbor_cell_num, pearson))
                except Exception:
                    continue

        if len(all_cell_similarities) == 0:
            skipped_count += 1
            continue

        cell_sim_dict = {}
        for cell_num, sim in all_cell_similarities:
            if cell_num not in cell_sim_dict or sim > cell_sim_dict[cell_num]:
                cell_sim_dict[cell_num] = sim

        sorted_cells = sorted(cell_sim_dict.items(), key=lambda x: -x[1])[:top_k]

        tad_results[tad_key] = {
            'top_k_cells': [cell_num for cell_num, _ in sorted_cells],
            'similarities': [float(sim) for _, sim in sorted_cells],
            'tad_info': {
                'chromosome': chrom,
                'start': int(start),
                'end': int(end),
                'bin_size': bin_num
            }
        }

        processed_count += 1

    elapsed = time.time() - start_time
    print(f"\n  TAD similarity computation done.")
    print(f"    Time: {elapsed/60:.1f} min")
    print(f"    Processed: {processed_count} TADs")
    print(f"    Skipped: {skipped_count} TADs")

    return tad_results


def main():
    print("=" * 80)
    print("Top-10 similar cells per TAD based on kNN")
    print("=" * 80)
    print(f"Config:")
    print(f"  Pairs dir: {PAIRS_DIR}")
    print(f"  kNN result dir: {SIMILAR_BASE_DIR}")
    print(f"  TAD file: {TAD_FILE}")
    print(f"  Resolution: {RESOLUTION}")
    print(f"  Top-K: {TOP_K}")
    print(f"  Neighbors used: {N_NEIGHBORS}")
    print("=" * 80)

    overall_start = time.time()

    print("\nStep 1: Load TAD boundaries")
    try:
        tad_df = pd.read_csv(TAD_FILE)
        print(f"  Loaded {len(tad_df)} TADs")
        if 'chromosome' in tad_df.columns:
            print(f"    Chromosome distribution: {tad_df['chromosome'].value_counts().to_dict()}")
    except Exception as e:
        print(f"  Failed to load TAD file: {e}")
        return 1

    chromosomes = [TARGET_CHROM]
    all_tad_results = {}

    for chrom in chromosomes:
        print(f"\n{'='*80}")
        print(f"Processing chromosome: {chrom}")
        print(f"{'='*80}")

        try:
            print(f"\nStep 2: Load kNN results")
            neighbor_indices, cell_ids = load_knn_results(SIMILAR_BASE_DIR)

            print(f"\nStep 3: Preload contacts on {chrom}")
            cell_contacts = preload_cell_chr1_contacts(
                cell_ids, PAIRS_DIR, chrom, RESOLUTION
            )

            print(f"\nStep 4: Compute TAD similarity")
            tad_results = calculate_tad_similarity_optimized(
                tad_df, neighbor_indices, cell_ids, cell_contacts,
                chrom, RESOLUTION, MAX_BIN_SIZE, MIN_BIN_SIZE,
                TOP_K, N_NEIGHBORS
            )

            all_tad_results.update(tad_results)
            print(f"  {chrom} done")

        except Exception as e:
            print(f"  {chrom} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*80}")
    print("Step 5: Save results")
    print(f"{'='*80}")

    if len(all_tad_results) == 0:
        print("  No TADs were successfully processed")
        return 1

    output_json = os.path.join(OUTPUT_DIR, "tad_top10_similar_cells.json")
    simple_results = {
        tad_key: data['top_k_cells']
        for tad_key, data in all_tad_results.items()
    }

    try:
        with open(output_json, "w") as f:
            json.dump(simple_results, f, indent=2)
        print(f"  Saved simplified results: {output_json}")
    except Exception as e:
        print(f"  Failed to save simplified results: {e}")

    output_json_full = os.path.join(OUTPUT_DIR, "tad_top10_similar_cells_full.json")
    try:
        with open(output_json_full, "w") as f:
            json.dump(all_tad_results, f, indent=2)
        print(f"  Saved full results: {output_json_full}")
    except Exception as e:
        print(f"  Failed to save full results: {e}")

    overall_elapsed = time.time() - overall_start
    print(f"\n  Summary:")
    print(f"    Total time: {overall_elapsed/60:.1f} min")
    print(f"    TADs processed: {len(all_tad_results)}")

    print(f"\n  Example results (first 3 TADs):")
    for i, (tad_key, data) in enumerate(list(all_tad_results.items())[:3]):
        print(f"    {tad_key}:")
        print(f"      Top-10 cells: {data['top_k_cells']}")
        print(f"      Similarities: {[f'{s:.3f}' for s in data['similarities']]}")

    print(f"\n{'='*80}")
    print("Done")
    print(f"{'='*80}")

    return 0


if __name__ == '__main__':
    exit(main())