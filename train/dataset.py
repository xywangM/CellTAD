#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch

from train.build_graph import (
    load_hic, align, diag_mask, ice_normalize, observed_over_expected,
    contact_distribution_features, compute_insulation_scores,
    compute_pseudobulk_tads, make_augmented_views, build_adjacency_paper,
)


class CellTADDataset:
    """Single-cell Hi-C training dataset: builds the graph, node features and views from the anchor and augmented maps."""

    def __init__(self, device, anchor_path, enhanced_paths, chrom, resolution, wd,
                 use_ice, feat_oe_clip, n_aug_views, aug_sigma,
                 tad_insul_window, n_pseudobulk_tads, min_tad_size,
                 backbone_plus_one, edge_oe_clip, adj_max_dist, agg_norm='sym'):
        self.device = device
        self.anchor_path = anchor_path
        self.enhanced_paths = enhanced_paths
        self.chrom = chrom
        self.resolution = resolution
        self.wd = wd
        self.use_ice = use_ice
        self.feat_oe_clip = feat_oe_clip
        self.n_aug_views = n_aug_views
        self.aug_sigma = aug_sigma
        self.tad_insul_window = tad_insul_window
        self.n_pseudobulk_tads = n_pseudobulk_tads
        self.min_tad_size = min_tad_size
        self.backbone_plus_one = backbone_plus_one
        self.edge_oe_clip = edge_oe_clip
        self.adj_max_dist = adj_max_dist
        self.agg_norm = agg_norm

        self.rng = np.random.default_rng(42)
        self._build()

    def _build(self):
        dev = self.device
        ch, rs = self.chrom, self.resolution

        anch = load_hic(self.anchor_path, ch, rs)
        enh = [load_hic(p, ch, rs) for p in self.enhanced_paths]
        am = align([anch] + enh)
        N = am[0].shape[0]
        mask = diag_mask(N, self.wd)
        print(f"  {N} bins, {len(am)} maps (1 anchor + {len(am)-1} enhanced candidates)")

        if self.use_ice:
            print("  ICE enabled -- for single cells this can amplify random contacts in low-coverage bins")
            am = [ice_normalize(m, mask) for m in am]
        enh = am[1:]

        print("  Computing O/E normalization (C_ij / E(d_ij))...")
        anchor_oe = observed_over_expected(am[0], mask, self.feat_oe_clip)
        enhanced_oe = [observed_over_expected(e, mask, self.feat_oe_clip) for e in enh]

        if self.n_aug_views > 0:
            if len(self.enhanced_paths) > 0:
                print("  Note: n_aug_views>0 with a non-empty enhanced_paths: the maps in "
                      "enhanced_paths are treated as a 'candidate pool' and used to regenerate "
                      "n_aug_views augmented maps via the insulation-based pseudo-bulk TAD split "
                      "+ random block replacement below, rather than being used directly as the "
                      "final views. If enhanced_paths already holds the final augmented maps "
                      "produced by augmentation_loader.py's real pipeline (TADGATE + correlation "
                      "ranking + per-TAD block replacement), this is probably not what you want -- "
                      "set n_aug_views to 0 to use enhanced_paths directly as the final views.")
            print("  Computing pseudo-bulk TAD blocks (paper uses TADGATE; insulation is used here "
                  "as a substitute, only for the internal augmentation branch above when n_aug_views>0)...")
            bulk = np.sum(np.stack(am, axis=0), axis=0)
            tad_labels, tad_boundaries = compute_pseudobulk_tads(
                bulk, self.tad_insul_window, self.n_pseudobulk_tads, self.min_tad_size)
            aug_oe = make_augmented_views(
                anchor_oe, enhanced_oe, tad_labels, self.n_aug_views,
                self.aug_sigma, mask, self.rng)
            all_oe = [anchor_oe] + aug_oe
            print(f"  Augmentation: 1 anchor + {len(aug_oe)} TAD-block-replaced "
                  f"(sigma={self.aug_sigma}) -> K={len(all_oe)} views")
        else:
            tad_labels, tad_boundaries = None, []
            all_oe = [anchor_oe] + enhanced_oe
            print(f"  Using the real augmented maps from enhanced_paths directly as the final views "
                  f"(internal insulation-based augmentation not triggered): "
                  f"1 anchor + {len(enhanced_oe)} precomputed augmented maps -> K={len(all_oe)} views")

        print("  Extracting node features (normalized contact distribution)...")
        feats = [torch.from_numpy(contact_distribution_features(oe, mask, self.wd)).float().to(dev)
                 for oe in all_oe]

        print("  Building adjacency matrix...")
        g = build_adjacency_paper(
            anchor_oe, mask,
            backbone_plus_one=self.backbone_plus_one,
            edge_oe_clip=self.edge_oe_clip,
            adj_max_dist=self.adj_max_dist,
            wd=self.wd,
            agg_norm=self.agg_norm,
            dev=dev,
        )

        insulation_scores = compute_insulation_scores(am[0])

        self.am = am
        self.mask = mask
        self.g = g
        self.feats = feats
        self.fd = feats[0].shape[1]
        self.N = N
        self.K_views = len(all_oe)
        self.insulation_scores = insulation_scores
        self.tad_labels = tad_labels
        self.tad_boundaries = tad_boundaries