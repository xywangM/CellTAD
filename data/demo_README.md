# CellTAD Demo Datasets

Two minimal demo datasets for running CellTAD end to end without preparing your own data or running the augmentation pipeline. Each contains **one anchor cell on `chr1` at 50 kb**, with Steps 0-7 of the augmentation pipeline already computed, so `--demo` goes straight to training and identification. All paths are relative to the repo root (`CellTAD/`).

| Demo | Folder | Anchor cell | Genome | Raw input |
|---|---|---|---|---|
| scMicro-C (GM12878) | `data/demo/GM-800U_006/demo/chr1/` | `GM-800U_006` | hg38 | pairs + `.cool` |
| Dip-C (mouse cortex) | `data/demo/cortex_p001_c001/demo/chr1/` | `GSM4382149_cortex-p001-cb_001` (`cell_001`) | mm10 | `.contacts.hic` |

## Download

The demo data are **not stored in the GitHub repository**. They are hosted externally at:

**Zenodo: https://doi.org/10.5281/zenodo.22859087**

Download the archive from the link above and unpack it into `data/` of your local copy of the repository:

```
cd CellTAD
wget https://zenodo.org/records/22859087/files/demo.tar.gz
tar -xzf demo.tar.gz -C data/
```

- The archive keeps the `data/demo/<anchor>/demo/chr1/...` structure shown under [Directory layout](#directory-layout); after unpacking, the folders `data/demo/GM-800U_006/` and `data/demo/cortex_p001_c001/` must exist.
- Size: about `xxx` MB. MD5 of `demo.tar.gz`: `xxx`.
- If you only need one demo, keep only its folder under `data/demo/`; `--demo` reads only the folder of the selected anchor.

## Usage

The demo is chosen by `--anchor-cell` (default: scMicro-C). Paths are resolved by `config.py` (`DEMO_DATASETS`, `apply_demo_mode()`, `resolve_paths()`).

```
cd CellTAD
conda activate CellTAD

# scMicro-C demo
python CellTAD.py --demo --run-identification --run-visualization

# Dip-C demo (uses data/chrom_mm10_sizes.txt unless --chrom-sizes is given)
python CellTAD.py --demo --anchor-cell GSM4382149_cortex-p001-cb_001 \
    --run-identification --run-visualization
```

- `--demo` cannot be combined with `--run-augmentation`, `--augmentation-only`, `--augmentation-start-from` or `--hic-dir`; an anchor without a bundled demo is an error.
- Without `--output-dir`, results go to `<demo>/demo/chr1/output/embedding` and `<demo>/demo/chr1/output/visualization/tad`. With `--output-dir X`, the figures and tables go to `X/visualization/tad`.
- The Dip-C anchor is a `.hic` file, so the Hi-C panel and the `hic_ratio` column of the identification step stay empty (a warning is printed); the TAD calls are not affected.

## Directory layout

```
data/demo/
├── GM-800U_006/demo/chr1/                       scMicro-C demo
│   ├── raw/
│   │   ├── GM-800U_006_allValidPairs.txt        anchor's valid pairs
│   │   └── GM-800U_006.cool                     anchor's binned map (view 0)
│   ├── pseudobulk/
│   │   ├── chr1_bulk_50000bp.npy                pseudo-bulk map
│   │   └── chr1_TADs.csv                        reference TADs
│   ├── embedding/
│   │   ├── total_decomp.npz                     scHiCluster embedding, all cells
│   │   └── cell_ids_in_order.txt                cell ids in row order
│   └── augmented/enhanced_top1..10.cool         10 augmented maps
└── cortex_p001_c001/demo/chr1/                  Dip-C demo
    ├── raw/
    │   └── GSM4382149_cortex-p001-cb_001.contacts.hic   anchor's contact map
    ├── pseudobulk/
    │   ├── chr1_bulk_50000bp.npy                pseudo-bulk map
    │   └── chr1_TADs.csv                        reference TADs
    ├── embedding/schicluster_embedding/
    │   ├── total_decomp.npz                     scHiCluster embedding, all cells
    │   └── cell_ids_in_order.txt                cell ids in row order
    └── augmented/enhanced_top1..10.cool         10 augmented maps
```

The two demos differ in one place: the scMicro-C embedding files sit directly in `embedding/`, the Dip-C ones in `embedding/schicluster_embedding/`. CellTAD uses `schicluster_embedding/` when that folder exists and `embedding/` otherwise.

The 10 augmented maps match `n_enhanced_candidates = 10`, i.e. 1 anchor + 10 augmented = 11 training views, so no extra option is needed.

## scMicro-C demo contents

| File | Contents |
|---|---|
| `raw/GM-800U_006_allValidPairs.txt` | valid pairs (v1.0) of the anchor cell, upper triangle, sorted chr1-chr2-pos1-pos2 |
| `raw/GM-800U_006.cool` | 4980 x 4980 bins (50 kb, chr1), 9,895,108 non-zero pixels |
| `pseudobulk/chr1_bulk_50000bp.npy` | pseudo-bulk contact matrix, chr1 at 50 kb |
| `pseudobulk/chr1_TADs.csv` | 458 reference TADs; columns `chromosome,start_pos,end_pos,TAD_id` |
| `embedding/total_decomp.npz` | key `arr_0`, shape (96, 49): one row per cell |
| `embedding/cell_ids_in_order.txt` | `GM-800U_001` ... `GM-800U_096`; the anchor is row 5 (0-based) |
| `augmented/enhanced_topN.cool` | same shape as the anchor; about 6.2 million non-zero pixels each (checked for `top1`-`top6`) |

## Dip-C demo contents

| File | Contents |
|---|---|
| `raw/GSM4382149_cortex-p001-cb_001.contacts.hic` | contact map of the anchor cell in `.hic` format; must contain the 50 kb resolution |
| `pseudobulk/chr1_bulk_50000bp.npy` | pseudo-bulk contact matrix, mm10 chr1 at 50 kb, square (3910 x 3910) |
| `pseudobulk/chr1_TADs.csv` | `xxx` reference TADs; columns `chromosome,start_pos,end_pos,TAD_id` |
| `embedding/schicluster_embedding/total_decomp.npz` | key `arr_0`, shape (`xxx`, `xxx`): one row per cell |
| `embedding/schicluster_embedding/cell_ids_in_order.txt` | the `xxx` cell ids in the same row order as `total_decomp.npz` |
| `augmented/enhanced_topN.cool` | `N` = 1 ... 10; each 3910 x 3910 bins at 50 kb, same shape as the anchor |

## Sanity checks

Run from the repo root.

```python
import cooler, numpy as np

# scMicro-C
base = "data/demo/GM-800U_006/demo/chr1"
c = cooler.Cooler(f"{base}/raw/GM-800U_006.cool")
assert c.binsize == 50000 and c.shape == (4980, 4980)
assert np.load(f"{base}/embedding/total_decomp.npz")["arr_0"].shape == (96, 49)
ids = open(f"{base}/embedding/cell_ids_in_order.txt").read().split()
assert ids[5] == "GM-800U_006"
bulk = np.load(f"{base}/pseudobulk/chr1_bulk_50000bp.npy")
assert bulk.shape == (4980, 4980)
for i in range(1, 11):
    ec = cooler.Cooler(f"{base}/augmented/enhanced_top{i}.cool")
    assert ec.binsize == 50000 and ec.shape == (4980, 4980)

# Dip-C (mouse chr1 at 50 kb has 3910 bins)
import hicstraw
base = "data/demo/cortex_p001_c001/demo/chr1"
hic = hicstraw.HiCFile(f"{base}/raw/GSM4382149_cortex-p001-cb_001.contacts.hic")
assert 50000 in [int(r) for r in hic.getResolutions()]
emb = f"{base}/embedding/schicluster_embedding"
ids = open(f"{emb}/cell_ids_in_order.txt").read().split()
assert np.load(f"{emb}/total_decomp.npz")["arr_0"].shape[0] == len(ids)
bulk = np.load(f"{base}/pseudobulk/chr1_bulk_50000bp.npy")
assert bulk.shape == (3910, 3910)
for i in range(1, 11):
    ec = cooler.Cooler(f"{base}/augmented/enhanced_top{i}.cool")
    assert ec.binsize == 50000 and ec.shape == (3910, 3910)
```
