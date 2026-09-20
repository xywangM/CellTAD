#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import scipy.sparse as sp
import torch
import dgl
import dgl.function as fn

try:
    import cooler
    HAS_COOLER = True
except Exception:
    HAS_COOLER = False

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from chrom_sizes import load_chrom_sizes

CS = load_chrom_sizes()


def load_pairs(path, chrom, res, cs):
    nb = cs // res + 1
    mat = np.zeros((nb, nb), dtype=np.float32)
    cv = [chrom, chrom.replace('chr', ''), 'chr' + chrom.replace('chr', '')]
    cnt = 0
    with open(path) as f:
        for ln in f:
            if ln[0] == '#' or not ln.strip():
                continue
            p = ln.strip().split('\t')
            if len(p) < 5:
                continue
            try:
                c1, p1, c2, p2 = p[1], int(p[2]), p[3], int(p[4])
                if c1 in cv and c2 in cv:
                    b1, b2 = p1 // res, p2 // res
                    if 0 <= b1 < nb and 0 <= b2 < nb:
                        mat[b1, b2] += 1
                        mat[b2, b1] += (b1 != b2)
                        cnt += 1
            except Exception:
                continue
    print(f"  pairs: {cnt:,}")
    return mat


def load_cool(p, c):
    return cooler.Cooler(p).matrix(balance=False).fetch(c).astype(np.float32)


def load_straw_hic(path, chrom, res):
    """Read a Juicer .hic into a dense symmetric raw-count matrix (chrom_size // res + 1 bins)."""
    import hicstraw

    hic_file = hicstraw.HiCFile(path)
    bare = chrom.replace('chr', '')
    wanted = {chrom, bare, 'chr' + bare}
    chrom_obj = next((c for c in hic_file.getChromosomes() if c.name in wanted), None)
    if chrom_obj is None:
        raise ValueError(f"chromosome {chrom} not found in {path}")

    records = hicstraw.straw('observed', 'NONE', path, chrom_obj.name, chrom_obj.name, 'BP', res)

    nb = (load_chrom_sizes(quiet=True).get(chrom) or chrom_obj.length) // res + 1
    rows, cols, vals = [], [], []
    for r_ in records:
        if r_.binX > r_.binY:
            continue
        i, j = int(r_.binX) // res, int(r_.binY) // res
        if i < nb and j < nb and r_.counts > 0:
            rows.append(i)
            cols.append(j)
            vals.append(r_.counts)
    m = sp.coo_matrix((vals, (rows, cols)), shape=(nb, nb), dtype=np.float32).toarray()
    m = m + m.T - np.diag(np.diag(m))
    print(f"  hic: {len(vals):,} non-zero bin pairs, {nb} bins")
    return m.astype(np.float32)


def load_hic(f, c, r):
    if f.endswith(('.txt', '.pairs')):
        return load_pairs(f, c, r, load_chrom_sizes(quiet=True).get(c, 0))
    if f.endswith('.hic'):
        return load_straw_hic(f, c, r)
    return load_cool(f, c)


def align(ms):
    s = min(m.shape[0] for m in ms)
    return [m[:s, :s] for m in ms]


def diag_mask(N, wd=100):
    wd = min(wd, N - 1)
    m = np.zeros((N, N), dtype=bool)
    m |= np.eye(N, dtype=bool)
    for i in range(1, wd + 1):
        d = np.ones(N - i, dtype=bool)
        m |= np.diag(d, k=i).astype(bool) | np.diag(d, k=-i).astype(bool)
    return m


def ice_normalize(mat, mask, n_iter=30, tol=1e-3):
    m = mat.astype(np.float64).copy()
    m[~mask] = 0.0
    for _ in range(n_iter):
        s = m.sum(axis=1)
        pos = s[s > 0]
        if pos.size == 0:
            break
        smean = pos.mean()
        b = s / (smean + 1e-12)
        b[b == 0] = 1.0
        m = m / b[:, None] / b[None, :]
        if np.abs(b - 1.0).mean() < tol:
            break
    m[~mask] = 0.0
    return m.astype(np.float32)


def observed_over_expected(mat, mask, clip=20.0):
    N = mat.shape[0]
    oe = np.zeros((N, N), dtype=np.float32)
    for d in range(0, N):
        idx_i = np.arange(0, N - d)
        idx_j = idx_i + d
        vals = mat[idx_i, idx_j]
        m = mask[idx_i, idx_j]
        valid = vals[m]
        E = valid.mean() if valid.size > 0 and valid.mean() > 0 else 0.0
        if E <= 0:
            continue
        oe_d = vals / E
        oe[idx_i, idx_j] = oe_d
        oe[idx_j, idx_i] = oe_d
    oe = np.clip(oe, 0.0, clip)
    oe[~mask] = 0.0
    return oe


def contact_distribution_features(oe, mask, wd):
    N = oe.shape[0]
    feat = np.zeros((N, 2 * wd + 1), dtype=np.float32)
    for i in range(N):
        for off in range(-wd, wd + 1):
            j = i + off
            if 0 <= j < N and mask[i, j]:
                feat[i, off + wd] = oe[i, j]
    s = feat.sum(axis=1, keepdims=True)
    s[s < 1e-8] = 1.0
    feat = feat / s
    return feat


def compute_insulation_scores(mat, window=5):
    N = mat.shape[0]
    ml = np.log1p(mat.astype(np.float32))
    scores = np.zeros(N, dtype=np.float32)
    for i in range(window, N - window):
        block = ml[i - window:i, i + 1:i + window + 1]
        scores[i] = block.mean() if block.size > 0 else 0.0
    return scores


def compute_pseudobulk_tads(bulk_mat, insul_window=5, target_n=96, min_tad_size=3):
    N = bulk_mat.shape[0]
    scores = compute_insulation_scores(bulk_mat, insul_window)
    valid = scores > 0
    minima = []
    for i in range(2, N - 2):
        if not valid[i]:
            continue
        if (scores[i] < scores[i - 1] and scores[i] < scores[i + 1] and
                scores[i] < scores[i - 2] and scores[i] < scores[i + 2]):
            minima.append((i, scores[i]))
    minima.sort(key=lambda x: x[1])
    n_bnd = min(len(minima), max(1, target_n - 1))
    boundaries = sorted([m[0] for m in minima[:n_bnd]])
    labels = np.zeros(N, dtype=np.int64)
    seg = 0
    bset = set(boundaries)
    for i in range(N):
        labels[i] = seg
        if i in bset:
            seg += 1
    n_tad = int(labels.max()) + 1
    sizes = np.bincount(labels)
    print(f"  Pseudo-bulk TADs: {n_tad} (boundaries: {len(boundaries)}), "
          f"size median={int(np.median(sizes))} min={int(sizes.min())} max={int(sizes.max())}")
    return labels, boundaries


def make_augmented_views(anchor_oe, candidate_oe_list, tad_labels, n_views,
                          sigma, mask, rng):
    n_lbl = int(tad_labels.max()) + 1
    tad_bins = [np.where(tad_labels == l)[0] for l in range(n_lbl)]
    views = []
    n_cand = len(candidate_oe_list)
    for t in range(n_views):
        aug = anchor_oe.copy()
        noise = rng.normal(0.0, sigma, size=aug.shape).astype(np.float32)
        noise[~mask] = 0.0
        aug = aug + noise
        aug = np.clip(aug, 0.0, None)
        for bins in tad_bins:
            if len(bins) == 0 or n_cand == 0:
                continue
            src = candidate_oe_list[rng.integers(0, n_cand)]
            s, e = bins.min(), bins.max() + 1
            aug[s:e, s:e] = src[s:e, s:e]
        aug[~mask] = 0.0
        views.append(aug)
    return views


def build_adjacency_paper(oe, mask, backbone_plus_one, edge_oe_clip, adj_max_dist,
                           wd, agg_norm='sym', dev='cpu'):
    N = oe.shape[0]
    src_list, dst_list, w_list = [], [], []
    raw_bb_list, raw_other_list = [], []
    n_d1, n_other = 0, 0
    plus = backbone_plus_one
    eclip = edge_oe_clip
    max_d = adj_max_dist if adj_max_dist > 0 else wd

    for i in range(N):
        lo = max(0, i - max_d)
        hi = min(N, i + max_d + 1)
        for j in range(lo, hi):
            if i == j:
                continue
            if not mask[i, j]:
                continue
            a_raw = float(oe[i, j])
            if a_raw <= 0 and abs(i - j) != 1:
                continue
            a = min(a_raw, eclip)
            if abs(i - j) == 1:
                w = a + plus
                n_d1 += 1
                raw_bb_list.append(a_raw)
            else:
                w = a
                n_other += 1
                raw_other_list.append(a_raw)
            src_list.append(i)
            dst_list.append(j)
            w_list.append(w)

    src = torch.tensor(src_list, dtype=torch.long)
    dst = torch.tensor(dst_list, dtype=torch.long)
    w = torch.tensor(w_list, dtype=torch.float32)

    g = dgl.graph((src, dst), num_nodes=N)
    g.edata['w'] = w
    ne_before = g.num_edges()
    g = dgl.add_self_loop(g)
    w_full = torch.cat([w, torch.ones(g.num_edges() - ne_before)])
    g.edata['w'] = w_full
    g = g.to(dev)

    agg_norm = agg_norm or 'sym'
    if agg_norm == 'sym':
        g.update_all(fn.copy_e('w', 'm'), fn.sum('m', 'deg'))
        dinv = g.ndata['deg'].clamp(min=1e-8).pow(-0.5)
        g.ndata['dinv'] = dinv
        g.apply_edges(fn.u_mul_v('dinv', 'dinv', 'nrm'))
        g.edata['w'] = g.edata['w'] * g.edata['nrm']
        for k in ('deg', 'dinv'):
            if k in g.ndata:
                del g.ndata[k]
        if 'nrm' in g.edata:
            del g.edata['nrm']

    src_np = np.asarray(src_list)
    dst_np = np.asarray(dst_list)
    w_np = np.asarray(w_list)
    bb_mask = np.abs(src_np - dst_np) == 1
    bb = w_np[bb_mask] if bb_mask.any() else np.array([0.0])
    ot = w_np[~bb_mask] if (~bb_mask).any() else np.array([0.0])

    raw_bb = np.asarray(raw_bb_list) if raw_bb_list else np.array([0.0])
    raw_other = np.asarray(raw_other_list) if raw_other_list else np.array([0.0])
    clipped_frac = float((raw_other > eclip).mean()) if raw_other.size else 0.0
    ratio = float(bb.mean() / max(ot.mean(), 1e-8))

    print(f"    [diag-clip] non-backbone edges, raw O/E: mean={raw_other.mean():.3f} "
          f"median={np.median(raw_other):.3f} p90={np.percentile(raw_other, 90):.3f} "
          f"max={raw_other.max():.3f}")
    print(f"    [diag-clip] fraction clipped by edge_oe_clip={eclip}: "
          f"{clipped_frac:.1%} (this much intra-TAD signal is clipped down to {eclip} before edges are built)")
    print(f"    [diag-clip] backbone edges, raw O/E: mean={raw_bb.mean():.3f} "
          f"median={np.median(raw_bb):.3f} (usually small; weight is mostly carried by +{plus})")
    print(f"    [diag] backbone final weight mean={bb.mean():.3f} vs other edges final weight mean={ot.mean():.3f} "
          f"| ratio={ratio:.2f}x (backbone should be larger; ratio closer to 1 with a higher clip "
          f"fraction suggests intra-TAD signal is more likely being systematically clipped by edge_oe_clip)")
    print(f"  Adjacency: {N} nodes, {g.num_edges()} edges "
          f"(|i-j|=1: {n_d1}, other: {n_other}, self-loops: {N}); avg degree={g.num_edges()/N:.1f}")
    print(f"    A_ij=C_ij/E(d_ij)(+{plus} when |i-j|=1); edge_oe_clip={eclip}, adj_max_dist={max_d}; "
          f"agg_norm='{agg_norm}'")
    return g