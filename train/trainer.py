#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import torch
from tqdm import tqdm

from model.ncla import loss_ncla_paper
from model.loss import combine_losses


def eval_quality(z):
    """Diagnostic metrics: ordering / mean+std of adjacent-bin distance / mean per-dim std."""
    with torch.no_grad():
        ad = torch.norm(z[:-1] - z[1:], p=2, dim=1)
        sk = torch.norm(z[:-2] - z[2:], p=2, dim=1)
        ordering = (ad[:-1] < sk).float().mean().item()
        d_mean = ad.mean().item()
        d_std = ad.std().item()
        std_dim_mean = z.std(dim=0).mean().item()
    return ordering, d_mean, d_std, std_dim_mean


class Trainer:
    def __init__(self, model, device, *,
                 num_epochs=220, warmup_epochs=30, lr=1e-3, weight_decay=1e-4,
                 grad_clip=5.0, checkpoint_metric='ncla', ckpt_smooth_evals=3,
                 alpha_max=0.7, alpha_ramp_epochs=10,
                 adj_std_guard_ratio=1.10, ord_guard_ratio=0.98,
                 ncla_tau=0.2, ncla_sample=512, ncla_r_local=2, ncla_neg_samples=128,
                 K_GIC=40, gic_tau=0.35, gic_z_scale_init=20.0,
                 eval_start=30, eval_freq=5, aggregate='concat'):
        self.model = model
        self.device = device

        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.checkpoint_metric = checkpoint_metric
        self.ckpt_smooth_evals = ckpt_smooth_evals
        self.alpha_max = alpha_max
        self.alpha_ramp_epochs = alpha_ramp_epochs
        self.adj_std_guard_ratio = adj_std_guard_ratio
        self.ord_guard_ratio = ord_guard_ratio
        self.ncla_tau = ncla_tau
        self.ncla_sample = ncla_sample
        self.ncla_r_local = ncla_r_local
        self.ncla_neg_samples = ncla_neg_samples
        self.K_GIC = K_GIC
        self.gic_tau = gic_tau
        self.gic_z_scale_init = gic_z_scale_init
        self.eval_start = eval_start
        self.eval_freq = eval_freq
        self.aggregate = aggregate

        self.logs = []
        self.best_epoch = 0
        self.baseline_adj_std = None
        self.baseline_ord = None

    def _alpha_at_epoch(self, ep):
        """Alpha schedule: 1.0 during warmup, then a linear ramp down to alpha_max."""
        warmup = self.warmup_epochs
        alpha_max = self.alpha_max
        ramp = self.alpha_ramp_epochs

        if ep < warmup:
            return 1.0
        if ramp <= 0:
            return alpha_max
        t = (ep - warmup) / float(ramp)
        t = max(0.0, min(1.0, t))
        return 1.0 + (alpha_max - 1.0) * t

    def _ckpt_locality_ok(self, cur_adj, cur_ord):
        """Locality guardrail: accept a checkpoint only if adj_std and Ord stay within the guard ratios of their baselines."""
        adj_guard_ratio = self.adj_std_guard_ratio
        ord_guard_ratio = self.ord_guard_ratio

        adj_ok = True
        if adj_guard_ratio > 0:
            if self.baseline_adj_std is None:
                self.baseline_adj_std = cur_adj
            else:
                adj_ok = cur_adj <= self.baseline_adj_std * adj_guard_ratio

        ord_ok = True
        if ord_guard_ratio > 0:
            if self.baseline_ord is None:
                self.baseline_ord = cur_ord
            else:
                ord_ok = cur_ord >= self.baseline_ord * ord_guard_ratio

        return adj_ok and ord_ok

    def fit(self, dataset):
        dev = self.device
        g = dataset.g
        feats = dataset.feats
        N = dataset.N
        warmup = self.warmup_epochs
        ckpt_metric = self.checkpoint_metric
        ckpt_smooth_evals = max(1, int(self.ckpt_smooth_evals))

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)

        self.logs = []
        t0 = time.time()
        best_score = 1e18
        best_state = None
        self.best_epoch = 0
        self.baseline_adj_std = None
        self.baseline_ord = None
        score_history = []

        pbar = tqdm(range(self.num_epochs), desc="train", ncols=180)
        for ep in pbar:
            self.model.train()
            embs = [self.model.encode(g, f) for f in feats]
            z_anchor = embs[0]
            z_views = embs[1:]

            L_ncla = loss_ncla_paper(
                z_anchor, z_views, tau=self.ncla_tau, sample=self.ncla_sample,
                r_local=self.ncla_r_local, n_neg=self.ncla_neg_samples)

            if ep == warmup and not self.model.gic.initialized:
                self.model.eval()
                with torch.no_grad():
                    z_init = self.model.encode(g, feats[0])
                self.model.gic.init_centroids(z_init)
                self.model.train()
                alpha_now = self._alpha_at_epoch(ep)
                print(f"\n  [GIC] E{ep}: KMeans init, {self.K_GIC} centroids "
                      f"(tau={self.gic_tau}, z_scale_init={self.gic_z_scale_init}, "
                      f"alpha_start={alpha_now:.3f} -> alpha_max={self.alpha_max} "
                      f"over {self.alpha_ramp_epochs} epochs, "
                      f"ckpt_smooth_evals={ckpt_smooth_evals}, "
                      f"adj_std_guard_ratio={self.adj_std_guard_ratio}, "
                      f"ord_guard_ratio={self.ord_guard_ratio})\n")

                diag = self.model.gic.diagnostics(z_init)
                uniform_r = 1.0 / self.K_GIC
                print(f"  [diag-1] h (GCN output) range: mean={diag['h_mean']:.3f} std={diag['h_std']:.3f} "
                      f"min={diag['h_min']:.3f} max={diag['h_max']:.3f}")
                print(f"  [diag-2] centroids (mu_k) range: mean={diag['c_mean']:.3f} std={diag['c_std']:.3f} "
                      f"min={diag['c_min']:.3f} max={diag['c_max']:.3f}")
                print(f"  [diag-3] pre-sigmoid (beta*r@centroids) range: mean={diag['logit_mean']:.3f} "
                      f"std={diag['logit_std']:.3f} min={diag['logit_min']:.3f} max={diag['logit_max']:.3f} "
                      f"(z_scale beta={diag['z_scale']:.2f})")
                print(f"  [diag-4] z (post-sigmoid) cross-node std mean={diag['z_std_over_nodes']:.5f} "
                      f"(near 0 => z has collapsed to a node-independent constant vector)")
                print(f"  [diag-5] sigmoid saturation fraction={diag['sigmoid_saturation_ratio']:.3f} "
                      f"(>0.5 => over half of (node, dim) pairs are saturated, gradient near dead)")
                print(f"  [diag-6] z computed independently per centroid, cross-cluster std mean={diag['z_std_over_clusters']:.5f} "
                      f"(near 0 => different cluster centroids already converge post-sigmoid)")
                print(f"  [diag-7] mean max soft-assignment weight r_max_mean={diag['r_max_mean']:.4f} "
                      f"(uniform baseline=1/K={uniform_r:.4f}; closer to baseline means flatter/still "
                      f"collapsing, closer to 1 means sharper/tau is working)\n")

            alpha = self._alpha_at_epoch(ep)
            if ep >= warmup and self.model.gic.initialized:
                perm = torch.randperm(N, device=dev)
                h_corrupt = self.model.encode(g, feats[0][perm])
                L_gic = self.model.gic(z_anchor, h_corrupt)
                gic_active = True
            else:
                L_gic = None
                gic_active = False

            total, L_gic_val = combine_losses(L_ncla, L_gic, alpha, gic_active, device=dev)

            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            opt.step()

            cur_ord = -1.0
            cur_adj = -1.0
            cur_std = -1.0
            if ep >= self.eval_start and ep % self.eval_freq == 0:
                self.model.eval()
                with torch.no_grad():
                    z_eval = self.model.encode(g, feats[0])
                cur_ord, cur_adj, cur_std, _ = eval_quality(z_eval)
                self.model.train()
                if ep >= warmup:
                    raw_score = total.item() if ckpt_metric == 'total' else L_ncla.item()
                    score_history.append(raw_score)
                    score = sum(score_history[-ckpt_smooth_evals:]) / len(score_history[-ckpt_smooth_evals:])
                    if score < best_score and self._ckpt_locality_ok(cur_adj, cur_ord):
                        best_score = score
                        self.best_epoch = ep
                        best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            self.logs.append({'ep': ep + 1, 'tot': float(total.item()),
                              'L_ncla': float(L_ncla.item()), 'L_gic': float(L_gic_val.item()),
                              'alpha': float(alpha),
                              'ordering': cur_ord if cur_ord >= 0 else None,
                              'adj_dist': cur_adj if cur_adj >= 0 else None})
            pbar.set_postfix({'T': f"{total.item():.3f}", 'NCLA': f"{L_ncla.item():.3f}",
                              'GIC': f"{L_gic_val.item():.3f}" if ep >= warmup else "-",
                              'a': f"{alpha:.2f}" if ep >= warmup else "-",
                              'Ord': f"{cur_ord:.3f}" if cur_ord >= 0 else "-"})

            if (ep + 1) % 40 == 0:
                el = time.time() - t0
                self.model.eval()
                with torch.no_grad():
                    z_eval = self.model.encode(g, feats[0])
                ord_s, dm, ds, std_d = eval_quality(z_eval)
                metric_name = 'best_total' if ckpt_metric == 'total' else 'best_NCLA'
                print(f"\n  E{ep+1} T={total.item():.3f} NCLA={L_ncla.item():.3f} "
                      f"GIC={L_gic_val.item():.3f} alpha={alpha:.3f} | Ord={ord_s*100:.1f}% "
                      f"adj={dm:.4f}+-{ds:.4f} std={std_d:.4f} "
                      f"{metric_name}={best_score:.4f}(smoothed,window={ckpt_smooth_evals})"
                      f"@E{self.best_epoch+1} (baseline_adj_std={self.baseline_adj_std}, "
                      f"baseline_ord={self.baseline_ord}) "
                      f"({el/60:.1f}m)\n")
                self.model.train()

        pbar.close()
        metric_name = 'best_total' if ckpt_metric == 'total' else 'best_NCLA'
        print(f"\nTraining done {(time.time()-t0)/60:.1f}m | "
              f"{metric_name}={best_score:.4f}(smoothed,window={ckpt_smooth_evals}) @ E{self.best_epoch+1} "
              f"(baseline_adj_std={self.baseline_adj_std}, adj_guard_ratio={self.adj_std_guard_ratio}, "
              f"baseline_ord={self.baseline_ord}, ord_guard_ratio={self.ord_guard_ratio})")

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()

        return self.logs

    def get_final_embedding(self, dataset):
        """After training, run every view once and aggregate per self.aggregate (paper: concat_k h_i^(k))."""
        self.model.eval()
        with torch.no_grad():
            view_embs = [self.model.encode(dataset.g, f).cpu().numpy() for f in dataset.feats]

        if self.aggregate == 'concat':
            import numpy as np
            final = np.concatenate(view_embs, axis=1)
        else:
            import numpy as np
            final = np.mean(np.stack(view_embs, axis=0), axis=0)

        print(f"  Aggregated ({self.aggregate}) over {len(view_embs)} views -> final {final.shape}")

        final_t = torch.from_numpy(final).to(self.device)
        ord_s, dm, ds, std_d = eval_quality(final_t)
        print(f"  Final | shape={final.shape} | ordering={ord_s*100:.2f}% "
              f"adj={dm:.4f}+-{ds:.4f} std={std_d:.4f}")

        return final, view_embs