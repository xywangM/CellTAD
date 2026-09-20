#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import numpy as np
import pandas as pd
import scipy.sparse as sp
import cooler

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config import get_default_config
from chrom_sizes import load_chrom_sizes

_cfg = get_default_config()

BASE_DIR = os.environ.get("CELLTAD_BASE_DIR", _cfg['base_dir'])
PAIRS_DIR = os.path.join(BASE_DIR, "GSE279583_extracted")

SCHIC_IMPUTE_DIR = os.path.join(BASE_DIR, "schicluster_impute")
SCHIC_COOL_DIR = os.path.join(BASE_DIR, "schicluster_cool")
EMBED_DIR = os.path.join(BASE_DIR, "embedding", "800U")

RESOLUTION = int(os.environ.get("CELLTAD_RESOLUTION", _cfg['resolution']))

CHROMOSOMES = [os.environ.get("CELLTAD_CHROM", _cfg['chrom'])]

CHROM_SIZE_FILE = os.path.join(BASE_DIR, "chrom_sizes_used.txt")

GENOME_CHROM_SIZES = load_chrom_sizes()

PAIRS_COL_CHROM1 = 1
PAIRS_COL_POS1 = 2
PAIRS_COL_CHROM2 = 3
PAIRS_COL_POS2 = 4

IMPUTE_PAD = 1
IMPUTE_STD = 1
IMPUTE_RP = 0.5
IMPUTE_TOL = 0.01

EMBED_DIM = 50
EMBED_DIST = 1_000_000
EMBED_SCALE_FACTOR = 100_000
EMBED_CPU = 4

PROJECT_ROOT = _PROJECT_ROOT
SCHICLUSTER_REPO_DIR = os.path.join(PROJECT_ROOT, "external", "scHiCluster")

if SCHICLUSTER_REPO_DIR not in sys.path:
    sys.path.insert(0, SCHICLUSTER_REPO_DIR)

# stub schicluster._version (repo is not pip-installed)
if 'schicluster._version' not in sys.modules:
    import types
    _fake_version_module = types.ModuleType('schicluster._version')
    _fake_version_module.version = '0.0.0-dev'
    sys.modules['schicluster._version'] = _fake_version_module

import schicluster
from schicluster.impute.impute_chromosome import impute_chromosome

try:
    from schicluster.embedding import embedding
except ImportError:
    from schicluster.embedding.calc_embedding import embedding


def get_chrom_sizes(chromosomes, size_lookup):
    """Build a chromosome-length Series from the size table (read from file), in the given order."""
    missing = [c for c in chromosomes if c not in size_lookup]
    if missing:
        raise ValueError(
            f"The chromosome size file has no length for: {missing}. "
            f"Check CELLTAD_CHROM_SIZES_FILE (genome) and CHROMOSOMES."
        )
    return pd.Series({c: size_lookup[c] for c in chromosomes}, name=0).astype(int)


def write_chrom_size_file_for_library(chrom_size_file, chromosomes, size_lookup):
    """Write the chromosome size file required by schicluster."""
    os.makedirs(os.path.dirname(chrom_size_file), exist_ok=True)
    with open(chrom_size_file, 'w') as f:
        for chrom in chromosomes:
            f.write(f"{chrom}\t{size_lookup[chrom]}\n")

    print(f"  Wrote chromosome size file for schicluster: {chrom_size_file} ({len(chromosomes)} chromosomes)")


def find_all_pairs_files(pairs_dir):
    pattern = os.path.join(pairs_dir, "*.allValidPairs.txt")
    return sorted(glob.glob(pattern))


def build_genome_bins(chrom_sizes, chromosomes, resolution):
    """Build the genome-wide bin table; returns (bins_df, {chrom: bin offset})."""
    rows = []
    offsets = {}
    cum = 0
    for chrom in chromosomes:
        chrom_len = int(chrom_sizes.loc[chrom])
        n_bins = chrom_len // resolution + 1
        offsets[chrom] = cum
        for b in range(n_bins):
            start = b * resolution
            end = min(start + resolution, chrom_len)
            rows.append((chrom, start, end))
        cum += n_bins
    bins_df = pd.DataFrame(rows, columns=['chrom', 'start', 'end'])
    return bins_df, offsets


def run_impute_chromosome_for_cell(pairs_file, cell_id, chrom, resolution,
                                    chrom_size_file, output_dir):
    """Run impute_chromosome() for one cell and chromosome; returns the .npz path or None."""
    chrom_dir = os.path.join(output_dir, chrom)
    os.makedirs(chrom_dir, exist_ok=True)
    output_path = os.path.join(chrom_dir, f"{cell_id}_{chrom}")

    try:
        impute_chromosome(
            chrom=chrom,
            resolution=resolution,
            output_path=output_path,
            contact_path=pairs_file,
            chrom_size_path=chrom_size_file,
            pad=IMPUTE_PAD,
            std=IMPUTE_STD,
            rp=IMPUTE_RP,
            tol=IMPUTE_TOL,
            chrom1=PAIRS_COL_CHROM1,
            pos1=PAIRS_COL_POS1,
            chrom2=PAIRS_COL_CHROM2,
            pos2=PAIRS_COL_POS2,
        )
    except Exception as e:
        print(f"      Imputation failed for {cell_id} / {chrom}: {e}")
        return None

    npz_path = output_path + ".npz"
    return npz_path if os.path.exists(npz_path) else None


def aggregate_cell_to_cooler(cell_id, chrom_npz_paths, bins_df, offsets, output_path):
    """Merge one cell's per-chromosome imputed matrices into a single .cool file."""
    bin1_list, bin2_list, val_list = [], [], []

    for chrom, npz_path in chrom_npz_paths.items():
        if npz_path is None or not os.path.exists(npz_path):
            continue
        mat = sp.load_npz(npz_path).tocoo()
        offset = offsets[chrom]
        r = mat.row + offset
        c = mat.col + offset
        mask = r <= c
        bin1_list.append(r[mask])
        bin2_list.append(c[mask])
        val_list.append(mat.data[mask])

    if not bin1_list:
        return None

    bin1 = np.concatenate(bin1_list)
    bin2 = np.concatenate(bin2_list)
    count = np.concatenate(val_list).astype(np.float32)

    pixels = pd.DataFrame({'bin1_id': bin1, 'bin2_id': bin2, 'count': count})
    pixels = pixels.groupby(['bin1_id', 'bin2_id'], as_index=False)['count'].sum()
    pixels = pixels.sort_values(['bin1_id', 'bin2_id']).reset_index(drop=True)

    cooler.create_cooler(
        cool_uri=output_path,
        bins=bins_df,
        pixels=pixels,
        dtypes={'count': 'float32'},
        ordered=True,
        symmetric_upper=True,
    )
    return output_path


def main():
    print("=" * 80)
    print("Module 1 - Step 3: get cell embeddings via schicluster library calls")
    print("=" * 80)

    os.makedirs(SCHIC_IMPUTE_DIR, exist_ok=True)
    os.makedirs(SCHIC_COOL_DIR, exist_ok=True)
    os.makedirs(EMBED_DIR, exist_ok=True)

    print("\nStep 3.0: Write the chromosome size file required by schicluster")
    write_chrom_size_file_for_library(CHROM_SIZE_FILE, CHROMOSOMES, GENOME_CHROM_SIZES)

    print("\nStep 3.1: Build chromosome lengths and the genome-wide bin table from the size table")
    chrom_sizes = get_chrom_sizes(CHROMOSOMES, GENOME_CHROM_SIZES)
    bins_df, offsets = build_genome_bins(chrom_sizes, CHROMOSOMES, RESOLUTION)
    print(f"  Genome-wide bin count: {len(bins_df)}")

    pairs_files = find_all_pairs_files(PAIRS_DIR)
    print(f"\n  Found {len(pairs_files)} single-cell pairs files")

    cell_cool_paths = {}

    for idx, pairs_file in enumerate(pairs_files, 1):
        cell_id = os.path.basename(pairs_file).replace(".allValidPairs.txt", "")
        print(f"\n[{idx}/{len(pairs_files)}] Processing cell: {cell_id}")

        chrom_npz_paths = {}
        for chrom in CHROMOSOMES:
            npz_path = run_impute_chromosome_for_cell(
                pairs_file, cell_id, chrom, RESOLUTION, CHROM_SIZE_FILE, SCHIC_IMPUTE_DIR
            )
            chrom_npz_paths[chrom] = npz_path

        n_ok = sum(1 for v in chrom_npz_paths.values() if v is not None)
        print(f"    Imputation succeeded for {n_ok}/{len(CHROMOSOMES)} chromosomes")

        if n_ok == 0:
            print(f"    No chromosome imputation succeeded for {cell_id}, skipping")
            continue

        cool_path = os.path.join(SCHIC_COOL_DIR, f"{cell_id}.cool")
        result = aggregate_cell_to_cooler(cell_id, chrom_npz_paths, bins_df, offsets, cool_path)
        if result is None:
            print(f"    Failed to aggregate {cell_id} into .cool, skipping")
            continue

        cell_cool_paths[cell_id] = cool_path
        print(f"    Generated: {cool_path}")

    if not cell_cool_paths:
        print("\nNo cell produced a .cool file, aborting.")
        return 1

    print(f"\nStep 3.4: Build cell_table ({len(cell_cool_paths)} cells)")
    cell_table_path = os.path.join(EMBED_DIR, "cell_table.tsv")
    with open(cell_table_path, 'w') as f:
        for cell_id, cool_path in cell_cool_paths.items():
            f.write(f"{cell_id}\t{cool_path}\n")
    print(f"  Saved: {cell_table_path}")

    cell_ids_in_order = list(cell_cool_paths.keys())
    cell_ids_path = os.path.join(EMBED_DIR, "cell_ids_in_order.txt")
    with open(cell_ids_path, 'w') as f:
        for cid in cell_ids_in_order:
            f.write(cid + "\n")
    print(f"  Saved cell_id list matching embedding row order: {cell_ids_path}")

    print("\nStep 3.5: Call schicluster.impute.embedding() to compute embeddings")
    embedding(
        cell_table_path=cell_table_path,
        output_dir=EMBED_DIR,
        chrom_size_path=CHROM_SIZE_FILE,
        dim=EMBED_DIM,
        dist=EMBED_DIST,
        resolution=RESOLUTION,
        scale_factor=EMBED_SCALE_FACTOR,
        cpu=EMBED_CPU,
        save_model=False,
        save_raw=False,
    )

    total_decomp_path = os.path.join(EMBED_DIR, "decomp", "total_decomp.npz")
    print("\n" + "=" * 80)
    print("scHiCluster embedding done")
    print(f"  Combined embedding: {total_decomp_path}")
    print(f"  Row-order cell_id list: {cell_ids_path}")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    exit(main())