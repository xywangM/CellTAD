#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import re
import subprocess

_THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_FILE_DIR, ".."))

SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "augmentation")

def _anchor_dir_name(anchor, base_dir):
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


def _hic_step(base_dir):
    """Step 0, only used for .hic input. Resumable, so 'done' is a marker written at its end."""
    return {
        "name": "0. Convert .hic to pairs",
        "script": "convert_hic_to_pairs.py",
        "cwd": SCRIPTS_DIR,
        "done": lambda: os.path.exists(os.path.join(base_dir, "hic_to_pairs.done")),
        "env_extra_key": "hic_dir",
    }


def _step_env(base_dir, chrom, resolution, chrom_sizes_file, force, anchor_cell=None):
    """Environment for every step script (they read CELLTAD_* variables)."""
    env = os.environ.copy()
    env["CELLTAD_BASE_DIR"] = base_dir
    if chrom:
        env["CELLTAD_CHROM"] = chrom
    if resolution:
        env["CELLTAD_RESOLUTION"] = str(resolution)
    if chrom_sizes_file:
        env["CELLTAD_CHROM_SIZES_FILE"] = (os.path.abspath(chrom_sizes_file)
                                           if os.path.exists(chrom_sizes_file) else chrom_sizes_file)
    if force:
        env["CELLTAD_FORCE"] = "1"
    if anchor_cell:
        env["CELLTAD_ANCHOR_CELL"] = str(anchor_cell)
    else:
        env.pop("CELLTAD_ANCHOR_CELL", None)
    return env


def _build_steps(base_dir, notebook_cwd, chrom=None, resolution=None, anchor_cell=None):
    chrom = chrom or "chr1"
    resolution = resolution or 50000

    def _exists(*parts):
        return os.path.exists(os.path.join(base_dir, *parts))

    anchor_dir = None
    if anchor_cell:
        anchor_dir = _anchor_dir_name(anchor_cell, base_dir)

    return [
        {
            "name": "1. Aggregate pseudo-bulk Hi-C",
            "script": "generate_pseudobulk.py",
            "cwd": SCRIPTS_DIR,
            "done": lambda: _exists("tadgate_results", "bulk_hic", f"{chrom}_bulk_{resolution}bp.npy"),
        },
        {
            "name": "2. TADGATE TAD boundary calling",
            "script": "run_TADGATE.py",
            "cwd": notebook_cwd,
            "done": lambda: _exists("tadgate_results", "tad_boundaries", f"{chrom}_TADs.csv"),
        },
        {
            "name": "3. scHiCluster cell embedding",
            "script": "generate_cell_embedding.py",
            "cwd": notebook_cwd,
            "done": lambda: (
                _exists("embedding", "800U", "decomp", "total_decomp.npz")
                and _exists("embedding", "800U", "cell_ids_in_order.txt")
            ),
        },
        {
            "name": "4. kNN similar cell search",
            "script": "find_neighbor_cells.py",
            "cwd": SCRIPTS_DIR,
            "done": lambda: _exists("similar", "similar_cells.pkl"),
        },
        {
            "name": "5. Top-10 similar cells per TAD",
            "script": "select_TAD_neighbors.py",
            "cwd": SCRIPTS_DIR,
            "done": lambda: _exists("similar", "tad_top10_similar_cells.json"),
        },
        {
            "name": "6. Generate augmented Hi-C maps",
            "script": "generate_augmented_contacts.py",
            "cwd": SCRIPTS_DIR,
            "done": (lambda: _exists("enhanced_maps", anchor_dir, chrom, "stats.json")) if anchor_dir
            else (lambda: _exists("enhanced_maps", "batch_processing_summary.json")),
        },
        {
            "name": "7. Convert npy to cool",
            "script": "convert_to_cool.py",
            "cwd": SCRIPTS_DIR,
            "done": (lambda: _exists("enhanced_maps", anchor_dir, chrom, "cool_conversion_report.json")) if anchor_dir
            else (lambda: _exists("enhanced_maps", "cool_conversion_report.json")),
        },
    ]


def run_step(step, force=False, env=None):
    name = step["name"]
    script_path = os.path.join(SCRIPTS_DIR, step["script"])

    print("\n" + "=" * 80)
    print(f"Step {name}")
    print(f"Script: {script_path}")
    print(f"cwd: {step['cwd']}")
    print("=" * 80)

    if not os.path.exists(script_path):
        print(f"Script not found: {script_path}")
        return False

    if not os.path.isdir(step["cwd"]):
        print(f"cwd not found: {step['cwd']} -- check the notebook_cwd argument passed by the caller")
        return False

    if not force and step["done"]():
        print("Output for this step already exists, skipping (use --force to rerun)")
        return True

    start = time.time()
    result = subprocess.run([sys.executable, script_path], cwd=step["cwd"], env=env)
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"Step failed (exit code {result.returncode}), elapsed {elapsed/60:.1f} min")
        return False

    if not step["done"]():
        print("Script exited normally (code 0) but the expected output file was not found. "
              "Check the script's internal logic or config.py's base_dir.")
        return False

    print(f"Done, elapsed {elapsed/60:.1f} min")
    return True


def run_augmentation_pipeline(base_dir, notebook_cwd=None, force=False,
                               start_from=1, only=None, verbose=True,
                               hic_dir=None, chrom_sizes_file=None,
                               chrom=None, resolution=None, anchor_cell=None):
    """Run or resume the augmentation pipeline; returns True on success."""
    notebook_cwd = notebook_cwd if notebook_cwd else SCRIPTS_DIR
    steps = _build_steps(base_dir, notebook_cwd, chrom, resolution, anchor_cell)
    hic_step = _hic_step(base_dir) if hic_dir else None
    env = _step_env(base_dir, chrom, resolution, chrom_sizes_file, force, anchor_cell)

    if only is not None:
        if only == 0:
            if hic_step is None:
                if verbose:
                    print("only=0 (.hic -> pairs) needs hic_dir")
                return False
            steps_to_run = [hic_step]
        elif 1 <= only <= len(steps):
            steps_to_run = [steps[only - 1]]
        else:
            if verbose:
                print(f"only must be between {0 if hic_step else 1} and {len(steps)}")
            return False
    else:
        if not (1 <= start_from <= len(steps)):
            if verbose:
                print(f"start_from must be between 1 and {len(steps)}")
            return False
        steps_to_run = steps[start_from - 1:]
        if hic_step is not None and start_from <= 1:
            steps_to_run = [hic_step] + steps_to_run

    if verbose:
        print("=" * 80)
        print("Augmentation data loader (augmentation_loader)")
        print("=" * 80)
        print(f"PROJECT_ROOT  = {PROJECT_ROOT}")
        print(f"SCRIPTS_DIR   = {SCRIPTS_DIR}")
        print(f"NOTEBOOK_CWD  = {notebook_cwd}")
        print(f"BASE_DIR      = {base_dir}")
        print(f"INPUT         = {'.hic from ' + hic_dir if hic_dir else 'pairs (no conversion step)'}")
        print(f"CHROM SIZES   = {chrom_sizes_file or '(default: data/chrom_hg38_sizes.txt / CELLTAD_CHROM_SIZES_FILE)'}")
        print(f"STEPS 6-7     = {'anchor only: ' + str(anchor_cell) if anchor_cell else 'all cells'}")
        print(f"Steps to run/check: {[s['name'] for s in steps_to_run]}")
        print("=" * 80)

    overall_start = time.time()

    for step in steps_to_run:
        step_env = dict(env)
        if step.get("env_extra_key") == "hic_dir":
            step_env["CELLTAD_HIC_DIR"] = hic_dir
        ok = run_step(step, force=force, env=step_env)
        if not ok:
            if verbose:
                print(f"\nPipeline stopped at '{step['name']}'. Fix the issue and rerun "
                      f"(completed steps are skipped automatically, no need to start over)")
            return False

    if verbose:
        total_elapsed = time.time() - overall_start
        print("\n" + "=" * 80)
        print(f"Augmented data ready. Elapsed this run: {total_elapsed/60:.1f} min")
        print("=" * 80)

    return True


def main():
    parser = argparse.ArgumentParser(description="Augmentation data loader (train/augmentation_loader.py)")
    parser.add_argument("--base-dir", type=str, default=None,
                         help="Override config.py's base_dir; uses the default config if omitted")
    parser.add_argument("--notebook-cwd", type=str, default=None,
                         help="Override config.py's augmentation_notebook_cwd; "
                              "defaults to augmentation/ itself if omitted")
    parser.add_argument("--force", action="store_true",
                         help="Ignore existing outputs and force-rerun every affected step")
    parser.add_argument("--start-from", type=int, default=1,
                         help="Step to start from (1-7); earlier steps are assumed done, default 1 "
                              "(with --hic-dir, Step 0 runs whenever this is 1)")
    parser.add_argument("--only", type=int, default=None,
                         help="Run only this one step (1-7; 0 = .hic -> pairs), ignoring the rest")
    parser.add_argument("--hic-dir", type=str, default=None,
                         help="Directory with *.hic files (.hic input). Omit for pairs input")
    parser.add_argument("--chrom-sizes", type=str, default=None,
                         help="Chromosome size table, e.g. data/chrom_mm10_sizes.txt (default: hg38)")
    parser.add_argument("--anchor-cell", type=str, default=None,
                         help="Generate augmented maps (Steps 6-7) only for this cell "
                              "(cell_XXX, GM-800U_XXX or the original .hic name); omit for all cells")
    parser.add_argument("--chrom", type=str, default=None, help="Override config.py's chrom")
    parser.add_argument("--resolution", type=int, default=None, help="Override config.py's resolution")
    args = parser.parse_args()

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from config import get_default_config
    default_cfg = get_default_config()

    base_dir = args.base_dir or default_cfg['base_dir']
    notebook_cwd = args.notebook_cwd or default_cfg.get('augmentation_notebook_cwd')

    ok = run_augmentation_pipeline(
        base_dir=base_dir, notebook_cwd=notebook_cwd,
        force=args.force, start_from=args.start_from, only=args.only, verbose=True,
        hic_dir=args.hic_dir or default_cfg.get('hic_dir'),
        chrom_sizes_file=args.chrom_sizes or default_cfg.get('chrom_sizes_file'),
        chrom=args.chrom, resolution=args.resolution,
        anchor_cell=args.anchor_cell,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())