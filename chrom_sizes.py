#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHROM_SIZES_FILE = os.path.join(_PROJECT_ROOT, "data", "chrom_hg38_sizes.txt")


def resolve_chrom_sizes_path(path=None):
    path = path or os.environ.get("CELLTAD_CHROM_SIZES_FILE") or DEFAULT_CHROM_SIZES_FILE
    if not os.path.isabs(path) and not os.path.exists(path):
        candidate = os.path.join(_PROJECT_ROOT, path)
        if os.path.exists(candidate):
            path = candidate
    return path


def load_chrom_sizes(path=None, quiet=False):
    path = resolve_chrom_sizes_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Chromosome size file not found: {path}\n"
            f"Expected layout: see CellTAD/data/chrom_hg38_sizes.txt / chrom_mm10_sizes.txt")
    sizes = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                sizes[parts[0]] = int(parts[1])
    if not quiet:
        print(f"  [chrom sizes] {path} ({len(sizes)} chromosomes)", flush=True)
    return sizes