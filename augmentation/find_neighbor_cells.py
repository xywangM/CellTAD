#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
import os
import sys
import pickle

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config import get_default_config

_cfg = get_default_config()

BASE_DIR = os.environ.get("CELLTAD_BASE_DIR", _cfg['base_dir'])
EMBED_DIR = os.path.join(BASE_DIR, "embedding", "800U")
OUTPUT_DIR = os.path.join(BASE_DIR, "similar")

K_NEIGHBORS = 10

TOTAL_DECOMP_FILE = os.path.join(EMBED_DIR, "decomp", "total_decomp.npz")
CELL_IDS_FILE = os.path.join(EMBED_DIR, "cell_ids_in_order.txt")


def load_cell_ids(cell_ids_file):
    """Read cell ids (one per line, in embedding row order)."""
    if not os.path.exists(cell_ids_file):
        print(f"  cell_id list file not found: {cell_ids_file}")
        return []

    with open(cell_ids_file, 'r') as f:
        cell_ids = [line.strip() for line in f if line.strip()]

    return cell_ids


def load_combined_embedding(total_decomp_file):
    """Load the genome-wide embedding from total_decomp.npz (sparse or dense layout)."""
    if not os.path.exists(total_decomp_file):
        raise FileNotFoundError(f"Embedding file not found: {total_decomp_file}")

    print(f"  Loading embedding: {total_decomp_file}")

    npz = np.load(total_decomp_file, allow_pickle=True)
    keys = list(npz.files)
    print(f"  Keys in npz: {keys}")

    sparse_required_keys = {'format', 'indices', 'indptr', 'data', 'shape'}
    if sparse_required_keys.issubset(set(keys)):
        print("  Detected scipy sparse matrix format, using scipy.sparse.load_npz")
        mat = sp.load_npz(total_decomp_file)
        embeddings = mat.toarray()
        return embeddings

    if len(keys) == 1:
        embeddings = npz[keys[0]]
        return np.asarray(embeddings)

    priority_keywords = ['decomp', 'embed', 'total', 'data', 'arr']
    for kw in priority_keywords:
        candidates = [k for k in keys if kw in k.lower()]
        if candidates:
            print(f"  Multiple arrays in npz, matched keyword '{kw}' -> key: {candidates[0]}")
            return np.asarray(npz[candidates[0]])

    raise KeyError(
        f"total_decomp.npz has multiple arrays and the embedding matrix "
        f"could not be identified automatically. Available keys: {keys}"
    )


def find_similar_cells(embeddings, k=10):
    """Return (indices, distances) of the k nearest neighbors per cell (Euclidean)."""
    n_cells = embeddings.shape[0]

    if n_cells <= k:
        k = max(1, n_cells - 1)
        print(f"  Not enough cells ({n_cells}), adjusting neighbor count to {k}")

    print(f"  Computing {n_cells}x{n_cells} distance matrix...")
    distance_matrix = cdist(embeddings, embeddings, metric='euclidean')

    neighbor_indices = np.zeros((n_cells, k), dtype=int)
    neighbor_distances = np.zeros((n_cells, k), dtype=float)

    for i in range(n_cells):
        distances = distance_matrix[i]
        sorted_indices = np.argsort(distances)

        neighbor_indices[i] = sorted_indices[1:k+1]
        neighbor_distances[i] = distances[sorted_indices[1:k+1]]

    print(f"  Found {k} nearest neighbors for each cell")

    return neighbor_indices, neighbor_distances


def save_results(neighbor_indices, neighbor_distances, cell_ids, output_dir, k):
    """Save kNN results."""
    os.makedirs(output_dir, exist_ok=True)

    indices_file = os.path.join(output_dir, "neighbor_indices.npy")
    distances_file = os.path.join(output_dir, "neighbor_distances.npy")

    np.save(indices_file, neighbor_indices)
    np.save(distances_file, neighbor_distances)

    results_dict = {
        'neighbor_indices': neighbor_indices,
        'neighbor_distances': neighbor_distances,
        'cell_ids': cell_ids,
        'k': k,
        'n_cells': len(cell_ids),
        'method': 'KNN (Euclidean distance on genome-wide combined embedding)'
    }

    pickle_file = os.path.join(output_dir, "similar_cells.pkl")
    with open(pickle_file, 'wb') as f:
        pickle.dump(results_dict, f)

    txt_file = os.path.join(output_dir, "similar_cells.txt")
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(f"# Similar cell search results (top-{k})\n")
        f.write(f"# Total cells: {len(cell_ids)}\n")
        f.write(f"# Method: Euclidean distance on genome-wide combined embedding\n")
        f.write(f"# Format: query_cell_id\tneighbor1_id:distance\tneighbor2_id:distance\t...\n\n")

        for i, cell_id in enumerate(cell_ids):
            neighbors = []
            for j in range(k):
                neighbor_idx = neighbor_indices[i, j]
                neighbor_id = cell_ids[neighbor_idx]
                distance = neighbor_distances[i, j]
                neighbors.append(f"{neighbor_id}:{distance:.6f}")

            f.write(f"{cell_id}\t" + "\t".join(neighbors) + "\n")

    stats_file = os.path.join(output_dir, "statistics.txt")
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Similar cell search statistics\n")
        f.write("=" * 70 + "\n\n")

        f.write("Basic info:\n")
        f.write(f"  Total cells: {len(cell_ids)}\n")
        f.write(f"  Neighbors per cell: {k}\n")
        f.write(f"  Similarity metric: Euclidean distance\n")
        f.write(f"  Embedding type: genome-wide combined embedding\n\n")

        f.write("Distance statistics:\n")
        f.write(f"  Min distance: {neighbor_distances.min():.6f}\n")
        f.write(f"  Max distance: {neighbor_distances.max():.6f}\n")
        f.write(f"  Mean distance: {neighbor_distances.mean():.6f}\n")
        f.write(f"  Median distance: {np.median(neighbor_distances):.6f}\n")
        f.write(f"  Std of distance: {neighbor_distances.std():.6f}\n\n")

        f.write("Average distance per neighbor rank:\n")
        for j in range(k):
            avg_dist = neighbor_distances[:, j].mean()
            f.write(f"  Neighbor {j+1}: {avg_dist:.6f}\n")

    print(f"  Results saved to: {output_dir}")


def main():
    print("=" * 80)
    print("Similar cell search via kNN on genome-wide combined embedding")
    print("=" * 80)
    print(f"Embedding dir: {EMBED_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Neighbor count: {K_NEIGHBORS}")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        print(f"\nStep 1: Load cell_id list")
        cell_ids = load_cell_ids(CELL_IDS_FILE)

        if not cell_ids:
            print(f"  Failed to load cell_id list, aborting")
            return 1

        print(f"  Loaded {len(cell_ids)} cell IDs")
        print(f"    Example: {cell_ids[:5]}")

        print(f"\nStep 2: Load genome-wide combined embedding")
        embeddings = load_combined_embedding(TOTAL_DECOMP_FILE)

        print(f"  Embedding loaded")
        print(f"    Shape: {embeddings.shape}")
        print(f"    Cell count: {embeddings.shape[0]}")
        print(f"    Embedding dim: {embeddings.shape[1]}")

        if len(cell_ids) != embeddings.shape[0]:
            print(f"  Warning: cell_id count ({len(cell_ids)}) does not match embedding count ({embeddings.shape[0]})")

            if len(cell_ids) < embeddings.shape[0]:
                print(f"     Extending cell_id list...")
                for i in range(len(cell_ids), embeddings.shape[0]):
                    cell_ids.append(f"cell_{i:04d}")
            else:
                print(f"     Truncating cell_id list...")
                cell_ids = cell_ids[:embeddings.shape[0]]

            print(f"     Adjusted cell count: {len(cell_ids)}")

        print(f"\nStep 3: Compute cell similarity")
        neighbor_indices, neighbor_distances = find_similar_cells(
            embeddings, k=K_NEIGHBORS
        )

        print(f"  Similar cell search done")
        print(f"    Mean distance: {neighbor_distances.mean():.6f}")
        print(f"    Distance range: [{neighbor_distances.min():.6f}, {neighbor_distances.max():.6f}]")

        print(f"\nStep 4: Save results")
        save_results(
            neighbor_indices, neighbor_distances, cell_ids,
            OUTPUT_DIR, neighbor_indices.shape[1]
        )

        print(f"\n  Example results (top-{K_NEIGHBORS} neighbors for the first 3 cells):")
        for i in range(min(3, len(cell_ids))):
            print(f"    [Cell {i+1}/{len(cell_ids)}: {cell_ids[i]}]")
            print(f"      Top-{K_NEIGHBORS} nearest neighbors:")
            for j in range(min(K_NEIGHBORS, neighbor_indices.shape[1])):
                neighbor_idx = neighbor_indices[i, j]
                neighbor_id = cell_ids[neighbor_idx]
                distance = neighbor_distances[i, j]
                print(f"        #{j+1:2d}. {neighbor_id:20s} (distance: {distance:.6f})")

    except FileNotFoundError as e:
        print(f"\n  File not found: {str(e)}")
        return 1

    except Exception as e:
        print(f"\n  Failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    print(f"\n{'='*80}")
    print("Similar cell computation done")
    print(f"{'='*80}")
    print(f"\nOutput dir: {OUTPUT_DIR}")
    print(f"  {OUTPUT_DIR}/")
    print(f"    |- neighbor_indices.npy")
    print(f"    |- neighbor_distances.npy")
    print(f"    |- similar_cells.pkl")
    print(f"    |- similar_cells.txt")
    print(f"    `- statistics.txt")

    return 0


if __name__ == "__main__":
    exit(main())