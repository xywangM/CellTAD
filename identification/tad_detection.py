#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


def compute_cosine_similarity_matrix(emb_region):
    norms = np.linalg.norm(emb_region, axis=1, keepdims=True) + 1e-8
    z_norm = emb_region / norms
    S = z_norm @ z_norm.T
    return S, z_norm


def compute_rho_insulation(S, flank=5):
    """Local insulation rho(i) = (L+R-X)/(L+R+X) over a flanking window of `flank` bins."""
    N = S.shape[0]
    rho = np.zeros(N, dtype=np.float64)
    for i in range(N):
        lw = min(flank, i)
        rw = min(flank, N - 1 - i)
        if lw < 1 or rw < 1:
            rho[i] = 0.0
            continue
        L = S[i - lw:i, i - lw:i].sum()
        R = S[i + 1:i + 1 + rw, i + 1:i + 1 + rw].sum()
        X = S[i - lw:i, i + 1:i + 1 + rw].sum()
        denom = L + R + X
        rho[i] = (L + R - X) / denom if abs(denom) > 1e-8 else 0.0
    return rho


def compute_delta_genomic(rho):
    N = len(rho)
    delta = np.full(N, float(N), dtype=np.float64)
    for i in range(N):
        dl = next((i - j for j in range(i - 1, -1, -1) if rho[j] > rho[i]), N)
        dr = next((j - i for j in range(i + 1, N) if rho[j] > rho[i]), N)
        delta[i] = min(dl, dr)
    return delta


def compute_gamma(rho, delta):
    """gamma(i) = rho(i) * delta(i)."""
    return rho * delta


def select_boundaries_by_decision_graph(gamma, edge_margin=0, min_gap_abs=0.0):
    """Select boundary bins by sorting gamma in descending order and cutting at the largest gap."""
    N = len(gamma)
    idx_pool = np.arange(edge_margin, N - edge_margin) if edge_margin > 0 else np.arange(N)
    if idx_pool.size < 2:
        return np.array([], dtype=int)

    order = idx_pool[np.argsort(-gamma[idx_pool])]
    sorted_vals = gamma[order]

    gaps = sorted_vals[:-1] - sorted_vals[1:]
    if gaps.size == 0:
        return np.array([], dtype=int)
    max_gap = gaps.max()
    if max_gap <= 0:
        return np.array([], dtype=int)

    if min_gap_abs > 0 and max_gap < min_gap_abs:
        return np.array([], dtype=int)

    k = int(np.argmax(gaps)) + 1
    boundary_bins = np.sort(order[:k])
    return boundary_bins.astype(int)


def select_boundaries_recursive(rho, min_tad_bins=3, edge_margin=0,
                                 min_split_bins=None, max_tad_bins=None,
                                 sub_edge_margin=None,
                                 recursive_min_gap_ratio=0.30,
                                 min_candidate_pool=None):
    """Apply the descending-gamma / largest-gap rule recursively to each sub-interval; returns global boundary_bins."""
    N = len(rho)
    if min_split_bins is None:
        min_split_bins = 2 * min_tad_bins
    if sub_edge_margin is None:
        sub_edge_margin = min_tad_bins
    if min_candidate_pool is None:
        min_candidate_pool = max(4, min_tad_bins)

    top_margin = min(sub_edge_margin, max(0, (N - 1) // 2)) if N > 1 else 0
    delta_full = compute_delta_genomic(rho)
    gamma_full = rho * delta_full
    if top_margin > 0 and N - 2 * top_margin >= 2:
        gamma_ref_pool = gamma_full[top_margin:N - top_margin]
    else:
        gamma_ref_pool = gamma_full
    if gamma_ref_pool.size >= 2:
        ref_scale = float(np.percentile(gamma_ref_pool, 90) - np.percentile(gamma_ref_pool, 10))
    else:
        ref_scale = 0.0
    min_gap_abs = recursive_min_gap_ratio * ref_scale if ref_scale > 1e-8 else 0.0

    def _recurse(lo, hi):
        length = hi - lo
        if length < min_split_bins:
            return []

        rho_sub = rho[lo:hi]
        delta_sub = compute_delta_genomic(rho_sub)
        gamma_sub = rho_sub * delta_sub

        max_margin_allowed = max(0, (length - min_candidate_pool) // 2)
        margin = min(sub_edge_margin, max_margin_allowed, max(0, (length - 1) // 2))

        local_bounds = select_boundaries_by_decision_graph(
            gamma_sub, edge_margin=margin, min_gap_abs=min_gap_abs)

        candidate = [b for b in local_bounds if 0 < b < length and gamma_sub[b] > 0]

        if not candidate and max_tad_bins is not None and length > max_tad_bins:
            lo_i = margin
            hi_i = length - margin
            if hi_i - lo_i >= 1:
                pos_idx = np.arange(lo_i, hi_i)
                pos_idx = pos_idx[gamma_sub[pos_idx] > 0]
                if pos_idx.size > 0:
                    best = pos_idx[np.argmax(gamma_sub[pos_idx])]
                    candidate = [int(best)]

        if not candidate:
            return []

        global_bounds = sorted(lo + b for b in candidate)
        edges = [lo] + global_bounds + [hi]
        all_bounds = list(global_bounds)
        for k in range(len(edges) - 1):
            s, e = edges[k], edges[k + 1]
            if e - s >= min_split_bins:
                all_bounds.extend(_recurse(s, e))
        return all_bounds

    lo0 = edge_margin if edge_margin > 0 else 0
    hi0 = N - edge_margin if edge_margin > 0 else N
    bounds = sorted(set(_recurse(lo0, hi0)))
    return np.array(bounds, dtype=int)


def segments_from_boundaries(boundary_bins, N):
    """Split the range into consecutive segments at the boundary bins, without merging."""
    bset = sorted(set(int(b) for b in boundary_bins if 0 < b < N))
    edges = [0] + bset + [N]
    segs = [[edges[k], edges[k + 1] - 1] for k in range(len(edges) - 1)]
    return segs


def _offdiag_stat(block, stat='mean'):
    """Mean or median of the off-diagonal entries of a square block."""
    n = block.shape[0]
    if n <= 1:
        return float(block[0, 0]) if block.size > 0 else 0.0
    mask = ~np.eye(n, dtype=bool)
    vals = block[mask]
    if vals.size == 0:
        return float(block[0, 0])
    return float(np.median(vals)) if stat == 'median' else float(vals.mean())


def _robust_intra(S, start, end, away_from_boundary, min_context=3, stat='median'):
    """Intra-segment similarity of segment [start, end], borrowing context bins from the far side when the segment is smaller than min_context."""
    size = end - start + 1
    if size >= min_context:
        block = S[start:end + 1, start:end + 1]
        return _offdiag_stat(block, stat=stat)

    need = min_context - size
    N = S.shape[0]
    if away_from_boundary < 0:
        new_start = max(0, start - need)
        new_end = end
    else:
        new_start = start
        new_end = min(N - 1, end + need)
    block = S[new_start:new_end + 1, new_start:new_end + 1]
    return _offdiag_stat(block, stat=stat)


def merge_similar_adjacent_segments(S, segs, merge_ratio_threshold=0.65,
                                     merge_stat='median', min_intra_context=3,
                                     debug=False):
    """Repeatedly merge adjacent segments whose cross/intra similarity ratio exceeds merge_ratio_threshold."""
    if len(segs) < 2:
        return [list(s) for s in segs]

    segs = [list(s) for s in segs]
    changed = True
    round_no = 0
    while changed and len(segs) >= 2:
        changed = False
        round_no += 1
        i = 0
        while i < len(segs) - 1:
            sA, eA = segs[i]
            sB, eB = segs[i + 1]
            block_cross = S[sA:eA + 1, sB:eB + 1]
            intra_A = _robust_intra(S, sA, eA, away_from_boundary=-1,
                                     min_context=min_intra_context, stat=merge_stat)
            intra_B = _robust_intra(S, sB, eB, away_from_boundary=+1,
                                     min_context=min_intra_context, stat=merge_stat)
            if block_cross.size > 0:
                cross = float(np.median(block_cross)) if merge_stat == 'median' else float(block_cross.mean())
            else:
                cross = 0.0
            denom = (intra_A + intra_B) / 2.0
            ratio = cross / denom if abs(denom) > 1e-8 else 1.0
            will_merge = ratio >= merge_ratio_threshold

            if debug:
                print(
                    f"[merge debug] round={round_no} "
                    f"A=[{sA},{eA}](size={eA - sA + 1}) "
                    f"B=[{sB},{eB}](size={eB - sB + 1}) | "
                    f"intra_A={intra_A:.4f} intra_B={intra_B:.4f} "
                    f"cross={cross:.4f} | ratio={ratio:.4f} "
                    f"vs threshold={merge_ratio_threshold:.4f} -> "
                    f"{'merge' if will_merge else 'keep boundary'}"
                )

            if will_merge:
                segs[i] = [sA, eB]
                del segs[i + 1]
                changed = True
            else:
                i += 1
    return segs


def boundaries_from_segments(segs):
    """Convert a segment list back to boundary bins (start of every segment except the first)."""
    if len(segs) < 2:
        return np.array([], dtype=int)
    return np.array([seg[0] for seg in segs[1:]], dtype=int)


def mark_gap_bins(z_norm, segs, dbscan_eps=0.3, dbscan_min=3, min_tad_bins=3,
                  return_breakdown=False):
    N = z_norm.shape[0]

    try:
        noise = (DBSCAN(eps=dbscan_eps, min_samples=dbscan_min,
                        metric='cosine').fit(z_norm).labels_ == -1)
    except Exception:
        noise = np.zeros(N, dtype=bool)
    gap = noise.copy()
    reason = {'noise': int(noise.sum()), 'tiny': 0}

    for s, e in segs:
        n = e - s + 1
        if n < min_tad_bins:
            idx = np.arange(s, e + 1)
            newly = int((~gap[idx]).sum())
            gap[idx] = True
            reason['tiny'] += newly

    if return_breakdown:
        return gap, reason
    return gap


def annotate_hic_quality(bounds, hic_sub, N, min_tad_bins=6):
    """Annotate boundaries with Hi-C ratios (for visualization only)."""
    annotations = []
    ml = np.log1p(hic_sub)
    for b in bounds:
        left_w = min(min_tad_bins, b)
        right_w = min(min_tad_bins, N - b)
        if left_w < 2 or right_w < 2:
            annotations.append((b, None)); continue
        left_int = ml[b - left_w:b, b - left_w:b].mean()
        right_int = ml[b:b + right_w, b:b + right_w].mean()
        cross = ml[b - left_w:b, b:b + right_w].mean()
        ratio = cross / ((left_int + right_int) / 2 + 1e-8)
        annotations.append((b, ratio))
    return annotations


def tad_param_grid_report(emb, start, end, hic_sub, candidates, base_kwargs=None,
                           hic_min_tad_bins=6):
    """Rerun detect_tads_paper for each candidate parameter set and print a ranking (diagnostic only)."""
    base_kwargs = dict(base_kwargs or {})
    N = end - start
    rows = []
    for cand in candidates:
        kwargs = dict(base_kwargs)
        kwargs.update(cand)
        tads, gap_mask, extra = detect_tads_paper(emb, start, end, **kwargs)
        hic_anno = annotate_hic_quality(
            list(tads['local_start']) if len(tads) else [],
            hic_sub, N, min_tad_bins=hic_min_tad_bins)
        ratios = [r for _, r in hic_anno if r is not None]
        sizes = tads['size_bins'].tolist() if len(tads) else []
        rows.append({
            'override': cand if cand else '(baseline=base_kwargs)',
            'n_tads': len(tads),
            'hic_ratio_mean': float(np.mean(ratios)) if ratios else None,
            'n_noise': int(extra['gap_reason']['noise']),
            'n_tiny': int(extra['gap_reason']['tiny']),
            'size_mean': float(np.mean(sizes)) if sizes else None,
            'size_std': float(np.std(sizes)) if sizes else None,
        })

    rows_sorted = sorted(
        rows, key=lambda r: (r['hic_ratio_mean'] is None,
                              r['hic_ratio_mean'] if r['hic_ratio_mean'] is not None else 0.0))

    print(f"  [diagnostic] TAD detection parameter candidates (ranked by hic_ratio_mean; noise/"
          f"tiny bin counts shown; adj_sim and Ordering are not included; not "
          f"adopted automatically, DETECT_KWARGS in main() decides)")
    print(f"  {'override':>48} {'n_tads':>7} {'hic_ratio_mean':>15} "
          f"{'n_noise':>8} {'n_tiny':>7} {'size_mean':>10} {'size_std':>9}")
    for r in rows_sorted:
        hr = f"{r['hic_ratio_mean']:.3f}" if r['hic_ratio_mean'] is not None else "  N/A"
        sm = f"{r['size_mean']:.2f}" if r['size_mean'] is not None else "N/A"
        ss = f"{r['size_std']:.2f}" if r['size_std'] is not None else "N/A"
        print(f"  {str(r['override']):>48} {r['n_tads']:>7} {hr:>15} "
              f"{r['n_noise']:>8} {r['n_tiny']:>7} {sm:>10} {ss:>9}")
    return rows_sorted


def boundary_margin_stats(z_norm, tads, window=5):
    """Return margin = sim(neighbor segment) - sim(own segment) records for bins near each TAD boundary."""
    records = []
    if tads is None or len(tads) < 2:
        return records
    t = tads.sort_values('local_start').reset_index(drop=True)

    for k in range(len(t) - 1):
        sA, eA = int(t.loc[k, 'local_start']), int(t.loc[k, 'local_end'])
        sB, eB = int(t.loc[k + 1, 'local_start']), int(t.loc[k + 1, 'local_end'])
        if eA < sA or eB < sB:
            continue
        blockA = z_norm[sA:eA + 1]
        blockB = z_norm[sB:eB + 1]
        if blockA.shape[0] == 0 or blockB.shape[0] == 0:
            continue

        centroid_A = blockA.mean(axis=0)
        centroid_B = blockB.mean(axis=0)
        cA = centroid_A / (np.linalg.norm(centroid_A) + 1e-8)
        cB = centroid_B / (np.linalg.norm(centroid_B) + 1e-8)

        wA = min(window, eA - sA + 1)
        for i in range(eA - wA + 1, eA + 1):
            sim_own = float(z_norm[i] @ cA)
            sim_nb = float(z_norm[i] @ cB)
            margin = sim_nb - sim_own
            records.append({'side': 'left', 'distance': eA - i + 1,
                             'margin': margin, 'more_like_neighbor': margin > 0})

        wB = min(window, eB - sB + 1)
        for i in range(sB, sB + wB):
            sim_own = float(z_norm[i] @ cB)
            sim_nb = float(z_norm[i] @ cA)
            margin = sim_nb - sim_own
            records.append({'side': 'right', 'distance': i - sB + 1,
                             'margin': margin, 'more_like_neighbor': margin > 0})
    return records


def boundary_margin_report(z_norm, tads, window=5):
    """Print boundary-margin records for bins near each TAD boundary (diagnostic only)."""
    if tads is None or len(tads) < 2:
        return
    t = tads.sort_values('local_start').reset_index(drop=True)

    print(f"  [diagnostic] boundary-bin membership at adjacent TAD boundaries "
          f"(up to {window} bins per side, original z_norm space, not UMAP; "
          f"margin=sim(neighbor)-sim(own), margin>0 means the bin is more like the "
          f"neighbor segment; for manual inspection only, no boundary is changed)")

    for k in range(len(t) - 1):
        sA, eA = int(t.loc[k, 'local_start']), int(t.loc[k, 'local_end'])
        sB, eB = int(t.loc[k + 1, 'local_start']), int(t.loc[k + 1, 'local_end'])
        idA, idB = int(t.loc[k, 'tad_id']), int(t.loc[k + 1, 'tad_id'])

        if eA < sA or eB < sB:
            continue
        blockA = z_norm[sA:eA + 1]
        blockB = z_norm[sB:eB + 1]
        if blockA.shape[0] == 0 or blockB.shape[0] == 0:
            continue

        centroid_A = blockA.mean(axis=0)
        centroid_B = blockB.mean(axis=0)
        cA = centroid_A / (np.linalg.norm(centroid_A) + 1e-8)
        cB = centroid_B / (np.linalg.norm(centroid_B) + 1e-8)

        print(f"    boundary T{idA + 1}[{sA},{eA}] | T{idB + 1}[{sB},{eB}]:")

        wA = min(window, eA - sA + 1)
        for i in range(eA - wA + 1, eA + 1):
            sim_own = float(z_norm[i] @ cA)
            sim_nb = float(z_norm[i] @ cB)
            margin = sim_nb - sim_own
            flag = " <- more like neighbor" if margin > 0 else ""
            print(f"      bin{i} (in T{idA + 1}, distance to boundary {eA - i + 1}): "
                  f"sim_own={sim_own:.3f} sim_neighbor={sim_nb:.3f} "
                  f"margin={margin:+.3f}{flag}")

        wB = min(window, eB - sB + 1)
        for i in range(sB, sB + wB):
            sim_own = float(z_norm[i] @ cB)
            sim_nb = float(z_norm[i] @ cA)
            margin = sim_nb - sim_own
            flag = " <- more like neighbor" if margin > 0 else ""
            print(f"      bin{i} (in T{idB + 1}, distance to boundary {i - sB + 1}): "
                  f"sim_own={sim_own:.3f} sim_neighbor={sim_nb:.3f} "
                  f"margin={margin:+.3f}{flag}")


def detect_tads_paper(emb, start, end, flank=5, min_tad_bins=3,
                      dbscan_eps=0.3, dbscan_min=3, edge_margin=0,
                      use_recursive_refinement=True, max_tad_bins=None,
                      sub_edge_margin=None, recursive_min_gap_ratio=0.30,
                      min_candidate_pool=None,
                      merge_similar_segments=True, merge_ratio_threshold=0.65,
                      merge_stat='median', min_intra_context=3,
                      merge_debug=False):
    """Detect TADs from an embedding: cosine similarity -> rho -> delta -> gamma -> boundaries -> segments -> gap annotation."""
    re_emb = emb[start:end]
    N = re_emb.shape[0]

    S, z_norm = compute_cosine_similarity_matrix(re_emb)
    rho = compute_rho_insulation(S, flank=flank)

    delta_display = compute_delta_genomic(rho)
    gamma_display = compute_gamma(rho, delta_display)

    if use_recursive_refinement:
        boundary_bins = select_boundaries_recursive(
            rho, min_tad_bins=min_tad_bins, edge_margin=edge_margin,
            max_tad_bins=max_tad_bins, sub_edge_margin=sub_edge_margin,
            recursive_min_gap_ratio=recursive_min_gap_ratio,
            min_candidate_pool=min_candidate_pool)
    else:
        boundary_bins = select_boundaries_by_decision_graph(gamma_display, edge_margin=edge_margin)
    segs = segments_from_boundaries(boundary_bins, N)

    if merge_similar_segments:
        segs = merge_similar_adjacent_segments(
            S, segs, merge_ratio_threshold=merge_ratio_threshold,
            merge_stat=merge_stat, min_intra_context=min_intra_context,
            debug=merge_debug)
        boundary_bins = boundaries_from_segments(segs)

    gap_mask, gap_reason = mark_gap_bins(
        z_norm, segs, dbscan_eps=dbscan_eps, dbscan_min=dbscan_min,
        min_tad_bins=min_tad_bins, return_breakdown=True)

    tads = []
    kept = 0
    for s, e in segs:
        if e < s:
            continue
        if gap_mask[s:e + 1].all():
            continue
        tads.append({
            'tad_id': kept, 'start_bin': s, 'end_bin': e,
            'local_start': s, 'local_end': e,
            'size_bins': e - s + 1,
            'center_bin': (s + e) // 2,
        })
        kept += 1
    tads_df = pd.DataFrame(tads)

    extra = {
        'rho': rho, 'delta': delta_display, 'gamma': gamma_display,
        'similarity_matrix': S, 'z_norm': z_norm,
        'boundary_bins': boundary_bins,
        'gap_reason': gap_reason,
    }
    return tads_df, gap_mask, extra