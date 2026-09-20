# CellTAD Demo Dataset — chr1, 50kb, anchor cell GM-800U_006

This is a minimal, real (subsampled) dataset for exercising the CellTAD
pipeline end-to-end without needing the full 96-cell scHi-C dataset. It
covers `chr1` at 50kb resolution and follows the anchor cell used in prior
runs, `GM-800U_006`.

This dataset matches the **new demo layout** consumed by
`config.py::apply_demo_mode()` / `resolve_paths()` (`cfg['demo_layout'] =
True`), which nests every file under `<anchor_cell>/demo/<chrom>/` rather
than directly under `<chrom>/` — see [Directory layout](#directory-layout)
below. All paths in this document are given **relative to the repo root**
(`CellTAD/`).

Verified contents (inspected with `cooler`/`numpy` before packaging):
| Stage in `augmentation_loader.py` | File | Contents |
|---|---|---|
| 0. Raw input | `data/demo/GM-800U_006/demo/chr1/raw/GM-800U_006_allValidPairs.txt` | Raw valid-pairs file (pairs format v1.0) for the anchor cell, upper-triangle, sorted chr1–chr2–pos1–pos2 |
| 0. Pairs → cool | `data/demo/GM-800U_006/demo/chr1/raw/GM-800U_006.cool` | Anchor cell's binned contact matrix. **4980 bins x 4980 bins, 50kb bins, chr1 only, 9,895,108 nonzero pixels** — this is view/K=0 (the un-augmented anchor graph) |
| 1. Pseudobulk → TADGATE | `data/demo/GM-800U_006/demo/chr1/pseudobulk/chr1_TADs.csv` | Pseudo-bulk TAD partition for chr1 (columns: `chromosome,start_pos,end_pos,TAD_id`; 458 TADs) — used to define the per-TAD augmentation windows |
| 1. Pseudobulk matrix | `data/demo/GM-800U_006/demo/chr1/pseudobulk/chr1_bulk_50000bp.npy` | Pseudo-bulk contact matrix for chr1 at 50kb resolution — corresponds to `cfg['pseudobulk_path']` in the demo layout (this file has no equivalent in the old production layout, see `config.py::resolve_paths()`) |
| 2. scHiCluster embedding | `data/demo/GM-800U_006/demo/chr1/embedding/total_decomp.npz` | Cell-embedding/decomposition matrix, key `arr_0`, shape **(96, 49)** — one row per cell, used for KNN neighbor-cell selection |
| — | `data/demo/GM-800U_006/demo/chr1/embedding/cell_ids_in_order.txt` | The 96 cell IDs (`GM-800U_001` … `GM-800U_096`) in the same row order as `total_decomp.npz`. Anchor cell `GM-800U_006` is **row index 5** (0-based) |
| 3–6. KNN neighbors → top-K correlation ranking → augmented contacts → npy-to-cool | `data/demo/GM-800U_006/demo/chr1/augmented/enhanced_top1.cool` … `enhanced_top10.cool` | 10 augmented views built from the anchor's structurally-similar neighbor cells, per-TAD top-correlation ranked and reconstructed back into cool format. Same shape as the anchor (4980 x 4980, 50kb bins); `top1`–`top6` verified at ~6.2M nonzero pixels each, `top7`–`top10` follow the same format |

This demo now ships the **full 10-view augmented set** (`enhanced_top1.cool`
… `enhanced_top10.cool`), matching production's default
`cfg['n_enhanced_candidates'] = 10` in `config.py::get_default_config()`
(1 anchor + 10 augmented = the K=11 views referenced in prior runs). No
`--num-views`-style override is needed to run the full pipeline against
this demo set.

## Directory layout

```
CellTAD/
└── data/
    └── demo/
        └── GM-800U_006/
            └── demo/
                └── chr1/
                    ├── raw/
                    │   ├── GM-800U_006_allValidPairs.txt
                    │   └── GM-800U_006.cool
                    ├── pseudobulk/
                    │   ├── chr1_TADs.csv
                    │   └── chr1_bulk_50000bp.npy
                    ├── embedding/
                    │   ├── cell_ids_in_order.txt
                    │   └── total_decomp.npz
                    └── augmented/
                        ├── enhanced_top1.cool
                        ├── enhanced_top2.cool
                        ├── ...
                        └── enhanced_top10.cool
```

The `<anchor_cell>/demo/<chrom>/` nesting (rather than a flat `<chrom>/`)
is what lets `config.py::apply_demo_mode()` support multiple demo cells or
chromosomes side by side later without renaming anything already shipped.

## Using this dataset

The simplest way to point CellTAD at this dataset is the `--demo` flag,
which calls `apply_demo_mode()` internally and resolves every path above
automatically:

```
cd CellTAD
conda activate CellTAD
python CellTAD.py --demo --run-identification --run-visualization
```

Or from Python, to inspect the resolved paths before running anything:

```python
from config import get_default_config, apply_demo_mode, resolve_paths

cfg = resolve_paths(apply_demo_mode(get_default_config()))
print(cfg['anchor_path'])        # data/demo/GM-800U_006/demo/chr1/raw/GM-800U_006_allValidPairs.txt
print(cfg['enhanced_paths'][0])  # data/demo/GM-800U_006/demo/chr1/augmented/enhanced_top1.cool
print(cfg['pseudobulk_path'])    # data/demo/GM-800U_006/demo/chr1/pseudobulk/chr1_bulk_50000bp.npy
```

## Quick sanity checks

Run from the repo root (`CellTAD/`) so the relative paths below resolve
correctly:

```python
import cooler, numpy as np

base = "data/demo/GM-800U_006/demo/chr1"

c = cooler.Cooler(f"{base}/raw/GM-800U_006.cool")
assert c.binsize == 50000 and c.shape == (4980, 4980)

d = np.load(f"{base}/embedding/total_decomp.npz")["arr_0"]
assert d.shape == (96, 49)

cell_ids = open(f"{base}/embedding/cell_ids_in_order.txt").read().split()
assert cell_ids[5] == "GM-800U_006"  # anchor cell row

bulk = np.load(f"{base}/pseudobulk/chr1_bulk_50000bp.npy")
assert bulk.shape[0] == bulk.shape[1]  # square pseudobulk contact matrix

for i in range(1, 11):
    ec = cooler.Cooler(f"{base}/augmented/enhanced_top{i}.cool")
    assert ec.binsize == 50000 and ec.shape == (4980, 4980)
```
