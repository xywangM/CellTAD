#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import umap
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from pathlib import Path
from sklearn.metrics import silhouette_score

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
for p in (_PROJECT_ROOT, _THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from identification.tad_detection import (
    detect_tads_paper, annotate_hic_quality, tad_param_grid_report,
    boundary_margin_report, boundary_margin_stats)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.unicode_minus': False, 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.grid': False})


def pp(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def load_embedding(edir, cell, chrom):
    edir = Path(edir)
    pre = f"{cell}_{chrom}"
    emb = np.load(edir / f'{pre}_embedding.npy')
    pp(f"✓ Embedding: {emb.shape}")
    return emb


def extract_hic_region(pairs_path, chrom, bs, be, res):
    sz = be - bs
    mat = np.zeros((sz, sz), dtype=np.float64)
    if not pairs_path or not os.path.exists(pairs_path):
        pp(f"⚠ Hi-C pairs file not found: {pairs_path} (Hi-C panel and hic_ratio will be empty)")
        return mat
    if str(pairs_path).lower().endswith(('.hic', '.cool', '.mcool')):
        pp(f"⚠ {pairs_path} is not a text pairs file; the Hi-C panel and hic_ratio need a pairs file "
           f"(chr1 pos1 chr2 pos2 in columns 2-5) and will be empty")
        return mat
    rs, re_ = bs * res, be * res
    cnt = 0
    with open(pairs_path) as f:
        for ln in f:
            if ln[0] == '#' or not ln.strip():
                continue
            p = ln.strip().split('\t')
            if len(p) < 5:
                continue
            try:
                c1, p1, c2, p2 = p[1], int(p[2]), p[3], int(p[4])
                if c1 != chrom or c2 != chrom:
                    continue
                if not (rs <= p1 < re_ and rs <= p2 < re_):
                    continue
                b1, b2 = p1 // res - bs, p2 // res - bs
                if 0 <= b1 < sz and 0 <= b2 < sz:
                    mat[b1, b2] += 1
                    mat[b2, b1] += (b1 != b2)
                    cnt += 1
            except Exception:
                continue
    if cnt == 0:
        pp("⚠ no contacts found in this region (check --chrom / --resolution / the pairs file)")
    else:
        pp(f"✓ {cnt:,} contacts")
    return mat


def build_tad_table(tads, region, chrom, sb, res, anno_dict):
    """One row per TAD: genomic coordinates (bp, end exclusive), global bin numbers (end inclusive), hic_ratio."""
    rows = []
    for _, t in tads.iterrows():
        ls, le = int(t['local_start']), int(t['local_end'])
        rows.append({
            'region': region,
            'chromosome': chrom,
            'tad_id': int(t['tad_id']) + 1,
            'start_bp': (sb + ls) * res,
            'end_bp': (sb + le + 1) * res,
            'start_bin': sb + ls,
            'end_bin': sb + le,
            'size_bins': int(t['size_bins']),
            'hic_ratio': anno_dict.get(ls),
        })
    return pd.DataFrame(rows, columns=['region', 'chromosome', 'tad_id', 'start_bp', 'end_bp',
                                       'start_bin', 'end_bin', 'size_bins', 'hic_ratio'])


def compute_umap(emb, start, end, nn=15, md=0.05, spread=2.0):
    """Unsupervised UMAP (no TAD labels)."""
    pp(f"Computing UMAP (n_neighbors={nn}, min_dist={md}, spread={spread}, "
       f"metric='cosine', unsupervised)...")
    re_emb = emb[start:end]
    na = min(nn, len(re_emb) - 1)
    reducer = umap.UMAP(n_neighbors=na, min_dist=md, spread=spread,
                         n_components=2, random_state=42, metric='cosine')
    return reducer.fit_transform(re_emb)


def adjacent_bin_similarity(emb, start, end):
    """Cosine similarity of adjacent bins in the original embedding space."""
    region = emb[start:end]
    norm = region / (np.linalg.norm(region, axis=1, keepdims=True) + 1e-8)
    sims = np.sum(norm[:-1] * norm[1:], axis=1)
    return sims


def tad_labels_for_scoring(tads, gap_mask, nb):
    """Convert the tads table and gap_mask into an integer label array (diagnostics only)."""
    labels = np.full(nb, -1, dtype=int)
    for _, t in tads.iterrows():
        s, e = int(t['local_start']), int(t['local_end'])
        if 0 <= s and e < nb:
            labels[s:e + 1] = int(t['tad_id'])
    if len(gap_mask) == nb:
        labels[gap_mask] = -1
    return labels


def umap_grid_search_report(emb, start, end, labels, candidates):
    """Print a silhouette ranking for candidate UMAP parameters (diagnostic only)."""
    valid_mask = labels != -1
    n_valid = int(valid_mask.sum())
    n_unique = len(set(labels[valid_mask].tolist())) if n_valid > 0 else 0

    pp(f"[diagnostic] UMAP candidate silhouette ranking "
       f"(valid points {n_valid}/{len(labels)}, {n_unique} TAD clusters; "
       f"⚠ the score uses TAD labels; do not use it to choose plotting parameters)")

    rows = []
    for nn, md, spread in candidates:
        umap_e = compute_umap(emb, start, end, nn=nn, md=md, spread=spread)
        score = None
        if n_unique >= 2 and n_valid >= 2:
            try:
                vals, counts = np.unique(labels[valid_mask], return_counts=True)
                if counts.min() >= 2:
                    score = float(silhouette_score(
                        umap_e[valid_mask], labels[valid_mask], metric='euclidean'))
            except Exception as ex:
                pp(f"  ⚠ (nn={nn}, md={md}, spread={spread}) scoring failed: {ex}")
        rows.append((nn, md, spread, score))

    rows.sort(key=lambda r: (-1e9 if r[3] is None else -r[3]))
    print(f"  {'n_neighbors':>12} {'min_dist':>10} {'spread':>8} {'silhouette':>12}")
    for nn, md, spread, score in rows:
        score_str = f"{score:.4f}" if score is not None else "  N/A"
        print(f"  {nn:>12} {md:>10} {spread:>8} {score_str:>12}")


def print_scale_match_grid(emb, sb, eb, hic, base_kwargs, flank_candidates,
                            umap_nn_ref, window):
    """Print hic_ratio and boundary-margin statistics for candidate flank values (diagnostic only)."""
    N = eb - sb
    print(f"  [diagnostic] flank x UMAP_NN(={umap_nn_ref}, reference only)"
          f" scale-match candidates (DETECT_KWARGS/UMAP_NN used for plotting are unchanged)")
    print(f"  {'flank':>6} {'n_tads':>7} {'hic_ratio_mean':>15} "
          f"{'pct_like_neighbor':>22}")
    for fk in flank_candidates:
        cand_kwargs = dict(base_kwargs)
        cand_kwargs['flank'] = fk
        t_cand, g_cand, d_cand = detect_tads_paper(emb, sb, eb, **cand_kwargs)
        hic_anno = annotate_hic_quality(
            list(t_cand['local_start']) if len(t_cand) else [], hic, N, min_tad_bins=6)
        ratios = [r for _, r in hic_anno if r is not None]
        hic_mean = float(np.mean(ratios)) if ratios else None
        recs = boundary_margin_stats(d_cand['z_norm'], t_cand, window=window)
        r1 = [r for r in recs if r['side'] == 'right' and r['distance'] == 1]
        pos_pct = 100.0 * sum(r['more_like_neighbor'] for r in r1) / len(r1) if r1 else float('nan')
        hr_str = f"{hic_mean:.3f}" if hic_mean is not None else "  N/A"
        pos_str = f"{pos_pct:.0f}%" if r1 else "  N/A"
        print(f"  {fk:>6} {len(t_cand):>7} {hr_str:>15} {pos_str:>22}")


def print_boundary_margin_aggregate(all_records):
    """Print boundary-margin statistics aggregated across regions."""
    if not all_records:
        return
    print(f"\n{'='*80}\n[aggregate] boundary margin statistics across regions ({len(all_records)} records, "
          f"{len({(r['side'], r['distance']) for r in all_records})} (side,distance) groups)"
          f"\n{'='*80}")
    for side in ('left', 'right'):
        for dist in (1, 2, 3):
            sub = [r for r in all_records if r['side'] == side and r['distance'] == dist]
            if not sub:
                continue
            n_pos = sum(r['more_like_neighbor'] for r in sub)
            mean_margin = sum(r['margin'] for r in sub) / len(sub)
            print(f"  side={side:>5} distance={dist}: "
                  f"like_neighbor={n_pos}/{len(sub)} ({100*n_pos/len(sub):.0f}%) "
                  f"mean_margin={mean_margin:+.3f}")
    print(f"{'='*80}")
    print("  Note: side=right, distance=1 is the first bin of each new TAD segment (adjacent to the boundary).")
    print("  A share clearly above 50% with mean_margin clearly above 0 indicates a systematic")
    print("  boundary shift; a share near 50% with mean_margin near 0 indicates that color mixing")
    print("  in the UMAP is boundary-specific rather than a uniform shift, so moving all boundaries")
    print("  by one bin will not help; compare the flank x UMAP_NN scale-match table above instead.")
    print("  Tuning recursive_min_gap_ratio / merge_ratio_threshold has no effect on this.")
    print(f"{'='*80}")


def plot_viz(hic, umap_e, tads, gap_mask, smb, emb_end, src, dpc_info,
             use_recursive, adj_sims, umap_nn, umap_md, umap_spread=1.0,
             save=None, chrom='chr1'):
    pp("Creating visualization...")
    nt = len(tads)
    if nt <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:nt]
    elif nt <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))[:nt]
    else:
        colors = plt.cm.hsv(np.linspace(0, 1, nt))

    nb = len(umap_e)
    ta = np.full(nb, -1, dtype=int)
    for i, (_, t) in enumerate(tads.iterrows()):
        s, e = int(t['local_start']), int(t['local_end'])
        if 0 <= s and e < nb:
            ta[s:e + 1] = i
    if len(gap_mask) == nb:
        ta[gap_mask] = -1

    boundary_bins = dpc_info.get('boundary_bins', np.array([], dtype=int))

    has_sim = 'similarity_matrix' in dpc_info
    has_rho = 'rho' in dpc_info
    has_gamma = 'gamma' in dpc_info
    has_adj = adj_sims is not None and len(adj_sims) > 0

    ncols = 1
    wrs = [1.2]
    if has_sim: ncols += 1; wrs.append(1.0)
    if has_rho: ncols += 1; wrs.append(0.20)
    if has_gamma: ncols += 1; wrs.append(0.20)
    if has_adj: ncols += 1; wrs.append(0.20)
    ncols += 1; wrs.append(1.1)

    fig = plt.figure(figsize=(min(50, 5.0 * ncols), 7.5))
    gs = fig.add_gridspec(1, ncols, width_ratios=wrs, wspace=0.25)
    axes = [fig.add_subplot(gs[i]) for i in range(ncols)]

    col_idx = 0
    ax_hic = axes[col_idx]; col_idx += 1
    ax_sim = axes[col_idx] if has_sim else None
    if has_sim: col_idx += 1
    ax_rho = axes[col_idx] if has_rho else None
    if has_rho: col_idx += 1
    ax_gamma = axes[col_idx] if has_gamma else None
    if has_gamma: col_idx += 1
    ax_adj = axes[col_idx] if has_adj else None
    if has_adj: col_idx += 1
    ax_umap = axes[col_idx]

    for sp in ax_hic.spines.values():
        sp.set_visible(True); sp.set_color('black'); sp.set_linewidth(2.5)
    hl = np.log1p(hic)
    vm = np.percentile(hl[hl > 0], 98) if np.any(hl > 0) else 1.0
    im = ax_hic.imshow(hl, cmap="Reds", aspect='auto', origin='upper',
                       interpolation='nearest', vmin=0, vmax=vm)
    cb = plt.colorbar(im, ax=ax_hic, shrink=0.7)
    cb.set_label('Log(Contact+1)', fontsize=9, fontweight='bold')
    nh = hic.shape[0]
    for i, (_, t) in enumerate(tads.iterrows()):
        s, e = int(t['local_start']), int(t['local_end'])
        if 0 <= s and e < nh and e >= s:
            ax_hic.add_patch(patches.Rectangle((s - 0.5, s - 0.5), e - s + 1, e - s + 1,
                lw=2.5, edgecolor=colors[i], facecolor='none'))
            if e - s > 3:
                cx = (s + e) / 2
                ax_hic.text(cx, cx, f'T{i+1}', ha='center', va='center', fontsize=9,
                    fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i],
                    edgecolor='black', linewidth=1.5, alpha=0.9))
    tp = np.linspace(0, nh - 1, 6)
    tl_labels = [f'{smb+(p/(nh-1))*(emb_end-smb):.1f}' for p in tp]
    ax_hic.set_xticks(tp); ax_hic.set_xticklabels(tl_labels, rotation=45)
    ax_hic.set_yticks(tp); ax_hic.set_yticklabels(tl_labels)
    ax_hic.set_xlabel('Position (Mb)', fontsize=11, fontweight='bold')
    ax_hic.set_ylabel('Position (Mb)', fontsize=11, fontweight='bold')
    ax_hic.set_title('Hi-C Contact Matrix', fontsize=11, fontweight='bold')

    if has_sim:
        for sp in ax_sim.spines.values():
            sp.set_visible(True); sp.set_color('black'); sp.set_linewidth(2.5)
        S = dpc_info['similarity_matrix']
        im2 = ax_sim.imshow(S, cmap='viridis', aspect='auto', origin='upper',
                            interpolation='nearest', vmin=-0.2, vmax=1.0)
        cb2 = plt.colorbar(im2, ax=ax_sim, shrink=0.7)
        cb2.set_label('Cosine Similarity', fontsize=9, fontweight='bold')
        nS = S.shape[0]
        for i, (_, t) in enumerate(tads.iterrows()):
            s, e = int(t['local_start']), int(t['local_end'])
            if 0 <= s and e < nS and e >= s:
                ax_sim.add_patch(patches.Rectangle((s - 0.5, s - 0.5), e - s + 1, e - s + 1,
                    lw=2.0, edgecolor='white', facecolor='none'))
        tps = np.linspace(0, nS - 1, 6)
        ax_sim.set_xticks(tps); ax_sim.set_xticklabels(tl_labels, rotation=45)
        ax_sim.set_yticks(tps); ax_sim.set_yticklabels(tl_labels)
        ax_sim.set_xlabel('Position (Mb)', fontsize=11, fontweight='bold')
        ax_sim.set_title('Embedding Similarity\nS = Z·Zᵀ', fontsize=11, fontweight='bold')

    def draw_side_panel(ax, values, title, xlabel):
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color('black'); sp.set_linewidth(2.5)
        vv = values.copy()
        vmax = vv.max() if vv.max() > 0 else 1.0
        vv = vv / (vmax + 1e-8)
        y_pos = np.arange(len(vv))
        vmean = vv.mean()
        bar_colors = ['#4477AA' if v >= vmean else '#CC6677' for v in vv]
        ax.barh(y_pos, vv, height=1.0, color=bar_colors, alpha=0.7)
        for b in boundary_bins:
            ax.axhline(y=int(b), color='red', linewidth=1.2, alpha=0.8, linestyle='--')
        ax.set_ylim(len(vv) - 0.5, -0.5)
        ax.set_xlabel(xlabel, fontsize=9, fontweight='bold')
        ax.set_yticks([]); ax.set_title(title, fontsize=11, fontweight='bold')

    if has_rho:
        draw_side_panel(ax_rho, dpc_info['rho'], 'ρ\n(insulation)', 'ρ')
    if has_gamma:
        gamma_title = 'γ\n(ρ×δ, decision score)' + ('\n[recursive]' if use_recursive else '\n[single-cut]')
        draw_side_panel(ax_gamma, dpc_info['gamma'], gamma_title, 'γ')

    if has_adj:
        for sp in ax_adj.spines.values():
            sp.set_visible(True); sp.set_color('black'); sp.set_linewidth(2.5)
        y_pos = np.arange(len(adj_sims)) + 0.5
        ax_adj.plot(adj_sims, y_pos, color='#33A02C', linewidth=1.2)
        ax_adj.fill_betweenx(y_pos, 0, adj_sims, color='#33A02C', alpha=0.2)
        for b in boundary_bins:
            ax_adj.axhline(y=int(b), color='red', linewidth=1.2, alpha=0.8, linestyle='--')
        ax_adj.set_xlim(min(0, adj_sims.min() - 0.05), 1.0)
        ax_adj.set_ylim(len(adj_sims), 0)
        ax_adj.set_xlabel('cos(bin_i, bin_i+1)', fontsize=9, fontweight='bold')
        ax_adj.set_yticks([])
        ax_adj.set_title(f'Adjacent-bin similarity\n(original space, not UMAP)\nmean={adj_sims.mean():.2f}',
                         fontsize=10, fontweight='bold')

    for sp in ax_umap.spines.values():
        sp.set_visible(True); sp.set_color('black'); sp.set_linewidth(2.5)
    ax_umap.plot(umap_e[:, 0], umap_e[:, 1], 'k--', alpha=0.2, lw=0.8, zorder=0)
    ax_umap.scatter(umap_e[:, 0], umap_e[:, 1], c='lightgray', s=40, alpha=0.2,
                    edgecolors='gray', linewidths=0.3, zorder=1)
    for i, (_, t) in enumerate(tads.iterrows()):
        m = ta == i
        if not m.any():
            continue
        pts = umap_e[m]
        ax_umap.scatter(pts[:, 0], pts[:, 1], c=[colors[i]], s=100, alpha=0.8,
            edgecolors='black', linewidths=1.2, marker='o',
            label=f'T{i+1}({int(t["size_bins"])}bins)', zorder=2)
    ax_umap.set_xlabel('UMAP1', fontsize=11, fontweight='bold')
    ax_umap.set_ylabel('UMAP2', fontsize=11, fontweight='bold')
    ax_umap.set_title(
        f'UMAP (cosine, nn={umap_nn}, md={umap_md}, spread={umap_spread}, unsupervised)\n'
        f'gray = gap bin', fontsize=11, fontweight='bold')

    h, l = ax_umap.get_legend_handles_labels()
    if h:
        leg = ax_umap.legend(h, l, loc='center left', bbox_to_anchor=(1.02, 0.5),
            frameon=True, fontsize=8, ncol=1 if len(h) <= 8 else 2,
            title='TAD Domains', title_fontsize=9)
        leg.get_frame().set_facecolor('white')
        leg.get_frame().set_edgecolor('black'); leg.get_frame().set_linewidth(1.5)

    plt.tight_layout()
    mode_tag = 'recursive DPC' if use_recursive else 'single-cut DPC'
    fig.suptitle(f'{chrom}: {smb:.1f}–{emb_end:.1f} Mb | {src} | paper-aligned {mode_tag} | {len(tads)} TADs',
                fontsize=13, y=0.98, fontweight='bold')
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight', facecolor='white')
        pp(f"Saved: {save}")
    plt.close()
    return fig


def main(edir=None, cell=None, chrom=None, res=None, hic_file=None, outdir=None, regions=None):
    USE_RECURSIVE_REFINEMENT = True
    MERGE_DEBUG = False

    DETECT_KWARGS = dict(
        flank=5, min_tad_bins=3,
        dbscan_eps=0.4, dbscan_min=3, edge_margin=0,
        use_recursive_refinement=USE_RECURSIVE_REFINEMENT,
        merge_debug=MERGE_DEBUG,
        recursive_min_gap_ratio=0.30,
        merge_ratio_threshold=0.65,
        merge_stat='median', min_intra_context=3,
    )

    PRINT_TAD_PARAM_GRID = True
    TAD_PARAM_GRID_CANDIDATES = [
        {},
        {'recursive_min_gap_ratio': 0.25},
        {'recursive_min_gap_ratio': 0.35},
        {'merge_ratio_threshold': 0.75},
        {'merge_ratio_threshold': 0.55},
        {'flank': 6},
        {'flank': 7},
        {'dbscan_eps': 0.3},
    ]

    UMAP_NN = 15
    UMAP_MD = 0.05
    UMAP_SPREAD = 2.0

    PRINT_GRID_REPORT = True
    UMAP_GRID_CANDIDATES = [
        (8, 0.02, 1.0), (25, 0.15, 1.0),
        (15, 0.05, 1.5), (15, 0.05, 2.0), (15, 0.10, 2.0),
        (20, 0.05, 2.0), (10, 0.05, 1.5), (10, 0.02, 2.0),
    ]

    PRINT_BOUNDARY_MARGIN = True
    BOUNDARY_MARGIN_WINDOW = 5

    PRINT_SCALE_MATCH_GRID = True
    SCALE_MATCH_FLANK_CANDIDATES = [5, 8, 10, 15]

    MIN_REGION_BINS = 20

    print("=" * 80)
    print("CellTAD/visualization/tad.py - single-TAD detection and visualization")
    print("=" * 80)
    print("  Pipeline: cosine S → ρ=(L+R-X)/(L+R+X) → δ(genomic distance) → γ=ρ×δ")
    if USE_RECURSIVE_REFINEMENT:
        print("  Boundaries: gamma in descending order + largest-gap cutoff, applied recursively to each sub-interval")
    else:
        print("  Boundaries: gamma in descending order + largest-gap cutoff (single global cut)")
    print("  gap: DBSCAN noise + too-small TADs (<3 bins)")
    if MERGE_DEBUG:
        print("  merge_debug: on")
    print(f"  DETECT_KWARGS: "
          f"dbscan_eps={DETECT_KWARGS['dbscan_eps']}, "
          f"recursive_min_gap_ratio={DETECT_KWARGS['recursive_min_gap_ratio']}, "
          f"merge_ratio_threshold={DETECT_KWARGS['merge_ratio_threshold']}, "
          f"flank={DETECT_KWARGS['flank']}")
    print(f"  UMAP: fixed parameters n_neighbors={UMAP_NN}, min_dist={UMAP_MD}, "
          f"spread={UMAP_SPREAD} (unsupervised, no TAD labels)")
    if PRINT_GRID_REPORT:
        print(f"  UMAP candidate ranking: on (print only, {len(UMAP_GRID_CANDIDATES)} candidates, "
              f"plotting parameters unchanged)")
    if PRINT_TAD_PARAM_GRID:
        print(f"  Detection parameter ranking: on (print only, "
              f"{len(TAD_PARAM_GRID_CANDIDATES)} candidates, DETECT_KWARGS unchanged)")
    if PRINT_BOUNDARY_MARGIN:
        print(f"  Boundary-bin membership diagnostic: on (up to {BOUNDARY_MARGIN_WINDOW} bins per side, "
              f"print only; an aggregate summary is printed at the end)")
    if PRINT_SCALE_MATCH_GRID:
        print(f"  flank x UMAP_NN scale-match diagnostic: on (candidate flank="
              f"{SCALE_MATCH_FLANK_CANDIDATES}, print only, "
              f"DETECT_KWARGS/UMAP_NN unchanged)")
    print("  Adjacent-bin similarity diagnostic: on (independent of UMAP)")
    print("=" * 80)

    if any(v is None for v in (edir, cell, chrom, res, hic_file, outdir, regions)):
        from config import get_default_config, resolve_paths
        cfg = resolve_paths(get_default_config())
        edir = cfg['output_dir'] if edir is None else edir
        cell = cfg['anchor_cell'] if cell is None else cell
        chrom = cfg['chrom'] if chrom is None else chrom
        res = cfg['resolution'] if res is None else res
        hic_file = cfg['anchor_path'] if hic_file is None else hic_file
        outdir = cfg['visualization_outdir'] if outdir is None else outdir
        regions = cfg['visualization_regions'] if regions is None else regions
    res = int(res)
    os.makedirs(outdir, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    emb = load_embedding(edir, cell, chrom)
    N = emb.shape[0]

    summary_rows = []
    all_margin_records = []
    all_tad_tables = []

    for smb, emb_end in regions:
        print(f"\n{'='*60}")
        print(f"Region: {smb}-{emb_end} Mb")
        print(f"{'='*60}")
        sb = int(smb * 1e6 / res); eb = min(int(emb_end * 1e6 / res), N)
        if eb - sb < MIN_REGION_BINS:
            pp(f"⚠ Skipping region {smb}-{emb_end} Mb: only {max(eb - sb, 0)} bins fall inside "
               f"{chrom} ({N} bins at {res} bp; at least {MIN_REGION_BINS} needed)")
            continue
        region_name = f'{smb}-{emb_end}Mb'
        pp(f"Region: {smb}-{emb_end} Mb, bins [{sb},{eb})")

        hic = extract_hic_region(hic_file, chrom, sb, eb, res)
        has_hic = bool(hic.any())
        src = f"Real Hi-C (res={res})" if has_hic else f"Hi-C unavailable (res={res})"

        tads, gap_mask, dpc_info = detect_tads_paper(emb, sb, eb, **DETECT_KWARGS)

        pp(f"✓ {len(tads)} TADs  ({len(dpc_info['boundary_bins'])} boundary candidates, "
           f"gap {int(gap_mask.sum())} bin = "
           f"noise {dpc_info['gap_reason']['noise']} + tiny {dpc_info['gap_reason']['tiny']})")
        if has_hic:
            hic_anno = annotate_hic_quality(list(tads['local_start']) if len(tads) else [],
                                            hic, eb - sb, min_tad_bins=6)
        else:
            hic_anno = []
        anno_dict = {b: r for b, r in hic_anno}
        ratios = [r for r in anno_dict.values() if r is not None]
        for _, t in tads.iterrows():
            s = int(t['local_start'])
            ratio_str = f" hic_ratio={anno_dict[s]:.2f}" if s in anno_dict and anno_dict[s] is not None else ""
            print(f"  T{int(t['tad_id'])+1}: [{int(t['local_start'])},{int(t['local_end'])}] "
                  f"{int(t['size_bins'])}bins center@{int(t['center_bin'])}{ratio_str}")

        tad_table = build_tad_table(tads, region_name, chrom, sb, res, anno_dict)
        tad_csv = os.path.join(outdir, f'tad_table_{chrom}_{smb:.1f}-{emb_end:.1f}Mb_{run_ts}.csv')
        tad_table.to_csv(tad_csv, index=False)
        all_tad_tables.append(tad_table)
        pp(f"✓ TAD table ({len(tad_table)} TADs): {tad_csv}")

        if PRINT_TAD_PARAM_GRID:
            tad_param_grid_report(emb, sb, eb, hic, TAD_PARAM_GRID_CANDIDATES,
                                   base_kwargs=DETECT_KWARGS)

        adj_sims = adjacent_bin_similarity(emb, sb, eb)
        pp(f"✓ Adjacent-bin similarity: mean={adj_sims.mean():.3f} min={adj_sims.min():.3f} "
           f"max={adj_sims.max():.3f}")
        boundary_bins = dpc_info.get('boundary_bins', np.array([], dtype=int))
        if len(boundary_bins) > 0:
            print("  [diagnostic] adjacent-bin similarity at each boundary (values close to mean/max support the chain-effect hypothesis):")
            for b in sorted(int(x) for x in boundary_bins):
                if 0 < b <= len(adj_sims):
                    print(f"    boundary@bin{b}: cos(bin{b-1},bin{b})={adj_sims[b-1]:.3f}")

        margin_recs = (boundary_margin_stats(dpc_info['z_norm'], tads, window=BOUNDARY_MARGIN_WINDOW)
                       if len(tads) else [])
        if PRINT_BOUNDARY_MARGIN:
            boundary_margin_report(dpc_info['z_norm'], tads, window=BOUNDARY_MARGIN_WINDOW)
            all_margin_records.extend(margin_recs)
        first_bin = [r for r in margin_recs if r['side'] == 'right' and r['distance'] == 1]
        margin_pct = (100.0 * sum(r['more_like_neighbor'] for r in first_bin) / len(first_bin)
                      if first_bin else None)
        margin_mean = float(np.mean([r['margin'] for r in first_bin])) if first_bin else None

        if PRINT_SCALE_MATCH_GRID:
            print_scale_match_grid(emb, sb, eb, hic, DETECT_KWARGS,
                                    SCALE_MATCH_FLANK_CANDIDATES,
                                    umap_nn_ref=UMAP_NN, window=BOUNDARY_MARGIN_WINDOW)

        nb = eb - sb
        labels = tad_labels_for_scoring(tads, gap_mask, nb)
        n_noise = int(dpc_info['gap_reason']['noise'])

        if PRINT_GRID_REPORT:
            umap_grid_search_report(emb, sb, eb, labels, UMAP_GRID_CANDIDATES)

        umap_e = compute_umap(emb, sb, eb, nn=UMAP_NN, md=UMAP_MD, spread=UMAP_SPREAD)

        sp = os.path.join(outdir, f'tad_viz_{smb:.1f}-{emb_end:.1f}Mb_{run_ts}.png')
        plot_viz(hic, umap_e, tads, gap_mask, smb, emb_end, src, dpc_info,
                 USE_RECURSIVE_REFINEMENT, adj_sims, UMAP_NN, UMAP_MD, UMAP_SPREAD, sp, chrom=chrom)
        print(f"\n✅ Region {smb}-{emb_end} Mb done. TADs: {len(tads)} | Saved: {sp}")

        summary_rows.append({
            'region': region_name, 'chromosome': chrom, 'n_tads': len(tads),
            'hic_ratio_mean': float(np.mean(ratios)) if ratios else None,
            'adj_sim_mean': float(adj_sims.mean()), 'n_noise': n_noise,
            'boundary_like_neighbor_pct': margin_pct, 'boundary_margin_mean': margin_mean,
        })

    print(f"\n{'='*80}")
    print("Summary of external metrics (independent of UMAP/visualization)")
    print(f"{'='*80}")
    print(f"  {'region':>10} {'n_tads':>7} {'hic_ratio_mean':>15} {'adj_sim_mean':>13} {'n_noise':>8}")
    for r in summary_rows:
        hr = f"{r['hic_ratio_mean']:.3f}" if r['hic_ratio_mean'] is not None else "N/A"
        print(f"  {r['region']:>10} {r['n_tads']:>7} {hr:>15} {r['adj_sim_mean']:>13.3f} {r['n_noise']:>8}")
    print(f"{'='*80}")

    if PRINT_BOUNDARY_MARGIN:
        print_boundary_margin_aggregate(all_margin_records)

    all_tads_df = pd.concat(all_tad_tables, ignore_index=True) if all_tad_tables else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    if summary_rows:
        all_csv = os.path.join(outdir, f'tad_table_{chrom}_all_regions_{run_ts}.csv')
        sum_csv = os.path.join(outdir, f'external_metrics_{chrom}_{run_ts}.csv')
        all_tads_df.to_csv(all_csv, index=False)
        summary_df.to_csv(sum_csv, index=False)
        pp(f"✓ All-regions TAD table: {all_csv}")
        pp(f"✓ External metrics summary: {sum_csv}")
    else:
        pp("⚠ No region was processed; check --chrom / --resolution and the region list "
           "(cfg['visualization_regions'], in Mb)")

    print("✅ All done.")
    print(f"{'='*80}")
    return all_tads_df, summary_df


if __name__ == "__main__":
    main()