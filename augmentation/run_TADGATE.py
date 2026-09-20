# -*- coding: utf-8 -*-

import os
import sys
import hashlib
import numpy as np
import pandas as pd
import torch

# newer PyTorch removed 'verbose' from ReduceLROnPlateau; TADGATE still passes it
_OrigReduceLROnPlateau = torch.optim.lr_scheduler.ReduceLROnPlateau


class _CompatReduceLROnPlateau(_OrigReduceLROnPlateau):
    def __init__(self, *args, **kwargs):
        kwargs.pop('verbose', None)
        super().__init__(*args, **kwargs)


torch.optim.lr_scheduler.ReduceLROnPlateau = _CompatReduceLROnPlateau

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from config import get_default_config
from chrom_sizes import load_chrom_sizes

_cfg = get_default_config()

BASE_DIR = os.environ.get("CELLTAD_BASE_DIR", _cfg['base_dir'])
BULK_HIC_DIR = os.path.join(BASE_DIR, "tadgate_results", "bulk_hic")
TAD_BOUNDARY_DIR = os.path.join(BASE_DIR, "tadgate_results", "tad_boundaries")
EMBED_CACHE_DIR = os.path.join(BASE_DIR, "tadgate_results", "embedding_cache")

TARGET_CHROM = os.environ.get("CELLTAD_CHROM", _cfg['chrom'])
RESOLUTION = int(os.environ.get("CELLTAD_RESOLUTION", _cfg['resolution']))
FORCE = os.environ.get("CELLTAD_FORCE", "0") == "1"

CHROM_SIZES = load_chrom_sizes()

TADGATE_REPO_DIR = os.path.join(PROJECT_ROOT, "external", "TADGATE")

if TADGATE_REPO_DIR not in sys.path:
    sys.path.insert(0, TADGATE_REPO_DIR)

import TADGATE
from TADGATE import Call_TADs as CT
from TADGATE import TADGATE_main

GRAPH_RADIUS = 2
SPLIT_SIZE = 'all'
LAYER_NODE1 = 500
LAYER_NODE2 = 100
LR = 0.001
WEIGHT_DECAY = 0.0001
NUM_EPOCH = 500
EMBED_ATTENTION = False
WEIGHT_USE = 'Fix'
WEIGHT_RATE = 0.3
WEIGHT_RANGE = int(10_000_000 / RESOLUTION)

BD_WEIGHT_LIST = [1.5, 1.5, 1, 1]
# use 'K-means' if rpy2/R mclust is not available
CLUSTER_METHOD = 'Mclust'
WINDOW_RANGE = 5_000_000
WD_CI = 5
WD_P = 5
DIST = 3
PVALUE_CUT = 0.05
EXP_LENGTH = 500_000
LENGTH_CUT = 3
CONTACT_FOLD_CUT = 2
ZERO_RATIO_CUT = 0.3

# mclust >= 6.1 builds column names from deparse(call$data); rpy2 inlines the matrix values into
# the call, so the name vector is longer than ncol(data) and Mclust() fails in `colnames<-`.
MCLUST_WRAPPER_R = """
Mclust <- local({
    orig <- mclust::Mclust
    function(data, verbose = FALSE, ...) {
        data <- as.matrix(data)
        if (is.null(colnames(data))) colnames(data) <- paste0("V", seq_len(ncol(data)))
        orig(data, verbose = verbose, ...)
    }
})
"""


def normalize_matrix_scale(mat_hic):
    """Rescale the Hi-C matrix values into the range TADGATE was trained for (1000-10000)."""
    mat_hic = mat_hic.astype(np.float64)
    max_val = np.max(mat_hic)
    if max_val <= 0:
        raise ValueError(
            "pseudo-bulk matrix max value <= 0 (all zeros or invalid values), "
            "the scaling loop would never terminate. Check the bulk hic "
            "aggregation output from the previous step."
        )
    while np.max(mat_hic) <= 1000:
        mat_hic = mat_hic * 10
    while np.max(mat_hic) > 10000:
        mat_hic = mat_hic * 0.1
    return mat_hic


def patch_and_check_mclust():
    """Install the Mclust wrapper and run a tiny Mclust call through rpy2 (fails fast, before training)."""
    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri as numpy2ri

    robjects.r(MCLUST_WRAPPER_R)
    numpy2ri.activate()
    robjects.r.library('mclust')
    test_mat = np.random.default_rng(0).normal(size=(60, 10)).astype(np.float32)
    res = robjects.r['Mclust'](test_mat, G=list(range(2, 5)))
    if len(np.array(res[13])) != 60:
        raise RuntimeError("Mclust check failed: unexpected result layout")
    print("  Mclust check passed")


def file_md5(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def embedding_signature(bulk_matrix_path):
    return {
        "version": 1,
        "chrom": TARGET_CHROM,
        "resolution": RESOLUTION,
        "bulk_md5": file_md5(bulk_matrix_path),
        "params": {
            "graph_radius": GRAPH_RADIUS, "split_size": SPLIT_SIZE,
            "layer_node1": LAYER_NODE1, "layer_node2": LAYER_NODE2,
            "lr": LR, "weight_decay": WEIGHT_DECAY, "num_epoch": NUM_EPOCH,
            "embed_attention": EMBED_ATTENTION, "weight_use": WEIGHT_USE,
            "weight_range": WEIGHT_RANGE, "weight_rate": WEIGHT_RATE,
        },
    }


def load_embedding_cache(cache_path, signature):
    if FORCE:
        print("  CELLTAD_FORCE=1, ignoring the embedding cache")
        return None
    if not os.path.exists(cache_path):
        return None
    try:
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(cache_path, map_location="cpu")
    except Exception as e:
        print(f"  Embedding cache unreadable, retraining: {e}")
        return None
    if payload.get("signature") != signature:
        print("  Embedding cache does not match the current matrix/parameters, retraining")
        return None
    return payload["result"]


def save_embedding_cache(cache_path, signature, result):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = cache_path + ".tmp"
    try:
        torch.save({"signature": signature, "result": result}, tmp_path)
        os.replace(tmp_path, cache_path)
        print(f"  Embedding cached: {cache_path}")
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        print(f"  Could not write the embedding cache (continuing): {e}")


def load_bulk_matrix(bulk_hic_dir, chrom, resolution):
    """Load the dense pseudo-bulk matrix generated in the previous step."""
    bulk_matrix_path = os.path.join(
        bulk_hic_dir, f"{chrom}_bulk_{resolution}bp.npy"
    )
    if not os.path.exists(bulk_matrix_path):
        raise FileNotFoundError(
            f"Pseudo-bulk matrix not found: {bulk_matrix_path}\n"
            f"Run the previous step (aggregate all cells' Hi-C maps into bulk hic) first"
        )
    mat_hic = np.load(bulk_matrix_path)
    return mat_hic, bulk_matrix_path


def run_tadgate_embedding(hic_mat_all, chr_size, resolution, target_chr_l, device):
    """Call TADGATE's embedding training directly."""
    print("  Calling TADGATE_main.TADGATE_for_embedding() ...")
    TADGATE_res_all = TADGATE_main.TADGATE_for_embedding(
        hic_mat_all, chr_size, resolution, GRAPH_RADIUS, SPLIT_SIZE, device,
        LAYER_NODE1, LAYER_NODE2, LR, WEIGHT_DECAY, NUM_EPOCH,
        embed_attention=EMBED_ATTENTION, weight_use=WEIGHT_USE,
        weight_range=WEIGHT_RANGE, weight_rate=WEIGHT_RATE,
        target_chr_l=target_chr_l
    )
    return TADGATE_res_all


def run_tadgate_call_tads(TADGATE_res_all, chr_size, resolution, target_chr_l):
    """Call TADGATE's TAD calling directly."""
    print("  Calling CT.TADGATE_call_TADs() ...")
    TADGATE_tads_all = CT.TADGATE_call_TADs(
        TADGATE_res_all, chr_size, resolution, BD_WEIGHT_LIST, CLUSTER_METHOD,
        WINDOW_RANGE, WD_CI, WD_P, DIST, PVALUE_CUT, EXP_LENGTH,
        length_cut=LENGTH_CUT, contact_fold_cut=CONTACT_FOLD_CUT,
        zero_ratio_cut=ZERO_RATIO_CUT, target_chr_l=target_chr_l
    )
    return TADGATE_tads_all


def extract_domain_tads(TADGATE_tads_all, chrom):
    """Return the type == 'domain' TADs of chrom as (chromosome, start_pos, end_pos, TAD_id)."""
    df_tad_res = TADGATE_tads_all[chrom]['TADs'][0]
    df_domain = df_tad_res[df_tad_res['type'] == 'domain'].copy()
    df_domain = df_domain.reset_index(drop=True)

    tad_df = pd.DataFrame({
        'chromosome': [chrom] * len(df_domain),
        'start_pos': df_domain['start_pos'].astype(int).values,
        'end_pos': df_domain['end_pos'].astype(int).values,
        'TAD_id': range(1, len(df_domain) + 1),
    })
    return tad_df


print("=" * 80)
print("Module 1 - Step 2: call TADGATE directly to identify TADs on pseudo-bulk Hi-C")
print("=" * 80)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"  Using device: {device}")

if CLUSTER_METHOD == 'Mclust':
    patch_and_check_mclust()

print("\nStep 2.1: Load pseudo-bulk matrix")
mat_hic, bulk_matrix_path = load_bulk_matrix(BULK_HIC_DIR, TARGET_CHROM, RESOLUTION)
print(f"  Loaded: {bulk_matrix_path}")
print(f"    Matrix shape: {mat_hic.shape}")

mat_hic = normalize_matrix_scale(mat_hic)
print(f"    Max value after scaling: {np.max(mat_hic):.2f}")

chr_size = {TARGET_CHROM: CHROM_SIZES[TARGET_CHROM]}
hic_mat_all = {TARGET_CHROM: mat_hic}
target_chr_l = [TARGET_CHROM]

print("\nStep 2.2: Run TADGATE embedding")
cache_path = os.path.join(EMBED_CACHE_DIR, f"{TARGET_CHROM}_{RESOLUTION}bp_tadgate_embedding.pt")
signature = embedding_signature(bulk_matrix_path)
TADGATE_res_all = load_embedding_cache(cache_path, signature)
if TADGATE_res_all is None:
    TADGATE_res_all = run_tadgate_embedding(hic_mat_all, chr_size, RESOLUTION, target_chr_l, device)
    save_embedding_cache(cache_path, signature, TADGATE_res_all)
else:
    print(f"  Loaded cached embedding: {cache_path}")

print("\nStep 2.3: Call TADGATE to identify TADs")
TADGATE_tads_all = run_tadgate_call_tads(TADGATE_res_all, chr_size, RESOLUTION, target_chr_l)

print("\nStep 2.4: Extract domain-type TADs and save")
tad_df = extract_domain_tads(TADGATE_tads_all, TARGET_CHROM)
print(f"  Identified {len(tad_df)} TADs (type == 'domain')")

os.makedirs(TAD_BOUNDARY_DIR, exist_ok=True)
output_csv = os.path.join(TAD_BOUNDARY_DIR, f"{TARGET_CHROM}_TADs.csv")
tad_df.to_csv(output_csv, index=False)

print(f"\n  TAD boundaries (with genomic coordinates) saved: {output_csv}")
print(f"    Columns: {list(tad_df.columns)}")