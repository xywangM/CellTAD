#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import sys
import glob
import math
import time
import queue
import multiprocessing as mp

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config import get_default_config
from chrom_sizes import load_chrom_sizes

_cfg = get_default_config()

BASE_DIR = os.environ.get("CELLTAD_BASE_DIR", _cfg['base_dir'])
PAIRS_DIR = os.path.join(BASE_DIR, "GSE279583_extracted")
HIC_DIR = os.environ.get("CELLTAD_HIC_DIR", _cfg.get('hic_dir'))

TARGET_CHROM = os.environ.get("CELLTAD_CHROM", _cfg['chrom'])
RESOLUTION = int(os.environ.get("CELLTAD_RESOLUTION", _cfg['resolution']))
CHROMS = [c.strip() for c in os.environ.get("CELLTAD_HIC_CHROMS", TARGET_CHROM).split(",") if c.strip()]

MIN_BIN_PAIRS = int(os.environ.get("CELLTAD_MIN_BIN_PAIRS", "1000"))
WORKER_TIMEOUT = int(os.environ.get("CELLTAD_WORKER_TIMEOUT", "300"))
FORCE = os.environ.get("CELLTAD_FORCE", "0") == "1"

MAP_FILE = os.path.join(BASE_DIR, "cell_id_map.tsv")
BAD_FILE = os.path.join(BASE_DIR, "hic_to_pairs_low_quality.tsv")


def _find_chrom(hic_file, chrom):
    """Match 'chr1' against the names stored in the .hic ('chr1' or '1')."""
    bare = chrom.replace("chr", "")
    wanted = {chrom, bare, "chr" + bare}
    for c in hic_file.getChromosomes():
        if c.name in wanted:
            return c
    return None


def read_hic_contacts(hic_path, chrom, resolution):
    """Return (chromosome length stored in the .hic, straw records) for chrom x chrom."""
    import hicstraw

    hic_file = hicstraw.HiCFile(hic_path)
    try:
        available = [int(r) for r in hic_file.getResolutions()]
        if resolution not in available:
            raise ValueError(f"resolution {resolution} not in this .hic (available: {available})")
    except AttributeError:
        pass

    chrom_obj = _find_chrom(hic_file, chrom)
    if chrom_obj is None:
        raise ValueError(f"chromosome {chrom} not found in {os.path.basename(hic_path)}")

    records = hicstraw.straw('observed', 'NONE', hic_path,
                             chrom_obj.name, chrom_obj.name, 'BP', resolution)
    return chrom_obj.length, records


def hic_to_pairs(hic_path, chroms, resolution, out_path, chrom_sizes):
    """Write hic_path as a pairs file. Returns (n_bin_pairs, n_contacts)."""
    half = resolution // 2
    n_bin_pairs = 0
    n_contacts = 0
    with open(out_path, "w") as out:
        for chrom in chroms:
            length, records = read_hic_contacts(hic_path, chrom, resolution)
            expected = chrom_sizes.get(chrom)
            if expected and length != expected:
                print(f"  WARNING: {chrom} is {length:,} bp in {os.path.basename(hic_path)} but "
                      f"{expected:,} bp in the chromosome size file -- wrong genome? "
                      f"(CELLTAD_CHROM_SIZES_FILE)", flush=True)
            for r in records:
                if r.binX > r.binY:
                    continue
                c = r.counts
                if not math.isfinite(c) or c <= 0:
                    continue
                k = int(round(c))
                if k <= 0:
                    continue
                out.write(f".\t{chrom}\t{int(r.binX) + half}\t{chrom}\t{int(r.binY) + half}\t+\t+\n" * k)
                n_bin_pairs += 1
                n_contacts += k
    return n_bin_pairs, n_contacts


def _worker(hic_path, chroms, resolution, tmp_path, chrom_sizes, q):
    try:
        q.put(("ok",) + hic_to_pairs(hic_path, chroms, resolution, tmp_path, chrom_sizes))
    except Exception as e:
        q.put(("error", f"{type(e).__name__}: {e}"))


def convert_one(hic_path, out_path, chroms, resolution, chrom_sizes):
    """Convert one .hic in a child process; returns ("ok", n_bin_pairs, n_contacts) or (status, reason)."""
    tmp_path = out_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(hic_path, chroms, resolution, tmp_path, chrom_sizes, q))
    p.start()
    p.join(WORKER_TIMEOUT)

    if p.is_alive():
        p.kill()
        p.join()
        _rm(tmp_path)
        return ("timeout", f"no result after {WORKER_TIMEOUT}s")

    try:
        msg = q.get(timeout=2)
    except queue.Empty:
        _rm(tmp_path)
        return ("crash", f"child process died (exit code {p.exitcode}; -11 = segfault in hicstraw)")

    if msg[0] != "ok":
        _rm(tmp_path)
    return msg


def _rm(path):
    if os.path.exists(path):
        os.remove(path)


def original_id(hic_path):
    """GSM4382149_cortex-p001-cb_001.contacts.hic -> GSM4382149_cortex-p001-cb_001"""
    name = os.path.basename(hic_path)
    name = re.sub(r"\.hic$", "", name)
    name = re.sub(r"\.contacts$", "", name)
    return name


def read_cell_map():
    rows = []
    if os.path.exists(MAP_FILE):
        with open(MAP_FILE) as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 2 and p[0] != "cell_id" and not line.startswith("#"):
                    rows.append((p[0], p[1], p[2] if len(p) > 2 else ""))
    return rows


def assign_cell_ids(hic_files):
    """Assign sequential ids (cell_001, ...) to .hic files; existing ids are never renumbered."""
    rows = read_cell_map()
    by_orig = {orig: (cid, hp) for cid, orig, hp in rows}
    next_num = max([int(cid.split("_")[1]) for cid, _, _ in rows if re.fullmatch(r"cell_\d+", cid)] + [0]) + 1

    id_of = {}
    for path in hic_files:
        orig = original_id(path)
        if orig in by_orig:
            id_of[orig] = by_orig[orig][0]
        else:
            id_of[orig] = f"cell_{next_num:03d}"
            by_orig[orig] = (id_of[orig], path)
            next_num += 1

    tmp = MAP_FILE + ".tmp"
    with open(tmp, "w") as f:
        f.write("cell_id\toriginal_id\thic_path\n")
        for orig, (cid, hp) in sorted(by_orig.items(), key=lambda kv: kv[1][0]):
            f.write(f"{cid}\t{orig}\t{hp}\n")
    os.replace(tmp, MAP_FILE)
    return id_of


def read_bad_cells():
    bad = {}
    if os.path.exists(BAD_FILE):
        with open(BAD_FILE) as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 3 and p[0] != "cell_id":
                    bad[p[0]] = p[2]
    return bad


def record_bad(cell_id, orig, reason):
    new = not os.path.exists(BAD_FILE) or os.path.getsize(BAD_FILE) == 0
    with open(BAD_FILE, "a") as f:
        if new:
            f.write("cell_id\toriginal_id\treason\n")
        f.write(f"{cell_id}\t{orig}\t{reason}\n")
        f.flush()


def run(hic_dir=None, pairs_dir=None):
    hic_dir = hic_dir or HIC_DIR
    pairs_dir = pairs_dir or PAIRS_DIR

    print("=" * 80, flush=True)
    print("Step 0: .hic -> pairs", flush=True)
    print("=" * 80, flush=True)
    if not hic_dir:
        print("Error: set CELLTAD_HIC_DIR to the directory holding the .hic files", flush=True)
        return 1
    hic_files = sorted(glob.glob(os.path.join(hic_dir, "*.hic")))
    if not hic_files:
        print(f"Error: no *.hic files in {hic_dir}", flush=True)
        return 1

    chrom_sizes = load_chrom_sizes()
    print(f"  hic dir   : {hic_dir}  ({len(hic_files)} files)", flush=True)
    print(f"  pairs dir : {pairs_dir}", flush=True)
    print(f"  chroms    : {CHROMS}   resolution: {RESOLUTION} bp", flush=True)
    print(f"  min bin pairs: {MIN_BIN_PAIRS}   worker timeout: {WORKER_TIMEOUT}s   force: {FORCE}", flush=True)

    os.makedirs(pairs_dir, exist_ok=True)
    id_of = assign_cell_ids(hic_files)

    if FORCE:
        _rm(BAD_FILE)
    bad = {} if FORCE else read_bad_cells()

    n_ok = n_skip_done = n_skip_bad = n_lowq = n_fail = 0
    t0 = time.time()
    for idx, path in enumerate(hic_files, 1):
        orig = original_id(path)
        cid = id_of[orig]
        out_path = os.path.join(pairs_dir, f"{cid}.allValidPairs.txt")

        if not FORCE:
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                n_skip_done += 1
                continue
            if cid in bad:
                n_skip_bad += 1
                continue

        print(f"[{idx}/{len(hic_files)}] {orig} -> {cid}", flush=True)
        res = convert_one(path, out_path, CHROMS, RESOLUTION, chrom_sizes)

        if res[0] != "ok":
            n_fail += 1
            print(f"    FAILED ({res[0]}): {res[1]}", flush=True)
            record_bad(cid, orig, f"{res[0]}: {res[1]}")
            continue

        n_bin_pairs, n_contacts = res[1], res[2]
        if n_bin_pairs < MIN_BIN_PAIRS:
            n_lowq += 1
            _rm(out_path + ".tmp")
            print(f"    skipped: only {n_bin_pairs} bin pairs (< {MIN_BIN_PAIRS})", flush=True)
            record_bad(cid, orig, f"low_quality: n_bin_pairs={n_bin_pairs}")
            continue

        os.replace(out_path + ".tmp", out_path)
        n_ok += 1
        if n_ok % 50 == 0 or n_ok <= 3:
            print(f"    ok: {n_bin_pairs} bin pairs, {n_contacts} contacts "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    print("\n" + "-" * 80, flush=True)
    print(f"  converted now : {n_ok}", flush=True)
    print(f"  already done  : {n_skip_done}", flush=True)
    print(f"  known bad, skipped: {n_skip_bad}   (CELLTAD_FORCE=1 to retry them)", flush=True)
    print(f"  low quality   : {n_lowq}", flush=True)
    print(f"  failed        : {n_fail}", flush=True)
    print(f"  id map        : {MAP_FILE}", flush=True)
    if n_lowq or n_fail or n_skip_bad:
        print(f"  excluded cells: {BAD_FILE}", flush=True)

    total_usable = len(glob.glob(os.path.join(pairs_dir, "*.allValidPairs.txt")))
    print(f"  usable pairs files in {pairs_dir}: {total_usable}", flush=True)
    if total_usable > 0:
        with open(os.path.join(BASE_DIR, "hic_to_pairs.done"), "w") as f:
            f.write(f"{total_usable}\n")
    return 0 if total_usable > 0 else 1


if __name__ == "__main__":
    sys.exit(run())