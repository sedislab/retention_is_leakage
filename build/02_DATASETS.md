# 02 — DATASETS

Four **required** public datasets, two **stretch**. They are chosen so that each one answers a
specific objection a reviewer will raise, not to pad a table.

| | Dataset | Role in the paper | Why it is in the set |
|---|---|---|---|
| **D1** | CIFAR-100 | class-IL workhorse, 10 and 20 tasks | the field's default protocol; comparability |
| **D2** | ImageNet-R | class-IL, 200 classes, hard | the standard benchmark for frozen-ViT continual learning |
| **D3** | CUB-200-2011 | fine-grained, ~30 images/class | small per-class *n* — the **best case for Gram inversion (H5)** |
| **D4** | Camelyon17-WILDS | domain-IL over **5 real hospitals** | a *natural* federation; kills "your clients are a Dirichlet artifact" |
| D5 | FMoW-WILDS | natural **temporal** drift (2002–2017) + 5 regions | stretch: real time, real geography |
| D6 | DomainNet (or Office-Home) | domain-IL, 6 domains | stretch: standard domain-IL comparability |

**D4 is the one that earns the paper its realism claim.** Camelyon17 clients are actual hospitals
with actual patient-derived data, which is the exact setting the FCL literature invokes in its
motivation and never evaluates. Do not cut it.

---

## 1. Rules for `scripts/get_data.py`

Two phases, always, because compute nodes may have no internet (`01_KODIAK.md §3`):

```
python -m p3fcl.cli data --fetch   --dataset X   # login node, network, writes raw/ + checksums
python -m p3fcl.cli data --prepare --dataset X   # compute node, no network, writes datasets/ + splits
```

Requirements:

1. **Resolve, then record.** URLs rot. Try the canonical URL, then the listed mirrors, and write the
   URL that *actually worked* plus its `sha256` and byte count into `data/MANIFEST.json`. Never hard-code
   a URL as the only path, and never silently fall back without recording the fallback.
2. **Idempotent and resumable.** `wget -c`, skip on checksum match, `.tmp` + atomic rename.
3. **Verify before extract.** A truncated 2 GB tar that extracts 80% of the way is the worst failure
   mode here because it produces a plausible-looking dataset with missing classes.
4. **Never `download=True` on the compute path.** Pass `download=False` to torchvision and WILDS;
   pre-stage everything.
5. **Record the license** for each dataset in the manifest. This goes in the paper's ethics section.
   All six datasets are public research releases, which is what keeps the project inside Kodiak's data
   policy (Protected data is not permitted on the cluster — `01_KODIAK.md §0`). If a dataset's license
   cannot be established, it does not go on the cluster.
6. After prepare, run `python -m p3fcl.cli data --audit` and have it print n_images, n_classes,
   class histogram min/median/max, image size distribution, and the natural-partition field where
   one exists. Eyeball it. A dataset you have not looked at is a dataset that will embarrass you.

### `data/MANIFEST.json` schema

```json
{
  "cifar100": {
    "url_used": "...", "mirrors_tried": ["..."], "sha256": "...", "bytes": 169001437,
    "license": "MIT-like, Krizhevsky 2009", "n_images": 60000, "n_classes": 100,
    "natural_partition_field": null,
    "splits": {"train": 40000, "test": 10000, "ref": 10000},
    "fetched_utc": "2026-09-14T…", "prepared_utc": "…", "prepare_source_hash": "…"
  }
}
```

---

## 2. D1 — CIFAR-100

- Source: `torchvision.datasets.CIFAR100`, canonical
  `https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz` (~169 MB).
- 60,000 32×32 images, 100 classes, 600 per class.
- **Streams.** Class-IL, 10 tasks × 10 classes and 20 tasks × 5 classes; class order shuffled by
  the stream seed (`streams.class_incremental_tasks` already does this). Clients: Dirichlet(β) with
  β ∈ {0.1, 0.5, 1.0, ∞}, 10 clients default. β = 0.5 is the headline.
- **Long horizon.** The T = 50 run for FIG09 uses 50 tasks × 2 classes. Accept that per-task accuracy
  is noisy at 2 classes and report it; the point of that run is the ε and leakage trends in T, not
  a clean accuracy curve.
- Resize to 224 with bicubic for the ViT. Upsampling 32→224 is standard in this literature and is
  also a *limitation* worth one honest sentence.

## 3. D2 — ImageNet-R

- Canonical: `https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar` (~2 GB).
  Mirror: HuggingFace `axiong/imagenet-r`. Repo: `github.com/hendrycks/imagenet-r`.
- 30,000 images, 200 ImageNet classes, renditions (art, cartoons, sketches, …).
- **Streams.** 10 tasks × 20 classes and 20 tasks × 10 classes — the L2P/DualPrompt/C²Prompt protocol,
  so numbers are comparable to published work.
- **Split it yourself**: ImageNet-R ships as one pool. Use a fixed 80/10/10 train/test/ref split,
  stratified by class, seeded, and record the indices in the manifest.
- **Caveat to state in the paper:** the ViT-B/16 IN21k backbone has seen ImageNet-21k, which overlaps
  these classes. Everyone in this literature has this problem; say so once, cite it, move on. Do not
  pretend it is not there — a reviewer will.

## 4. D3 — CUB-200-2011

- Source: CaltechDATA record `https://data.caltech.edu/records/65de6-vp158`, file `CUB_200_2011.tgz`
  (~1.1 GB). The old `vision.caltech.edu/visipedia-data/...` URL is dead — try it, expect failure,
  record which mirror worked. TFDS (`caltech_birds2011`) is an acceptable fallback.
- 11,788 images, 200 species, **~30 images per class per split**.
- **Streams.** 10 tasks × 20 classes.
- **This is the H5 dataset.** Per-class *n* of ~30 puts the Gram-inversion attack in exactly the
  regime where the toy study said recovery degrades (exact at n ≤ 4, collapsing by n = 16 on
  isotropic synthetic features). Real ViT features are strongly anisotropic and class-clustered, so
  the effective dimension is far below 768 and the adversary may do considerably better than the
  toy — **but that is a prediction, not a result.** The deliverable is the curve wherever it lands.
- Use the official train/test split file; carve `ref` out of train, stratified.

## 5. D4 — Camelyon17-WILDS

- Source: `pip install wilds`, then `wilds.get_dataset(dataset="camelyon17", download=True, root_dir=...)`
  on the **login node**. ~10 GB, ~455,000 96×96 histopathology patches.
- **5 hospitals** as the natural domain/client field. Binary label (tumour / normal).
- **Streams.** Domain-IL: hospitals arrive as a sequence of domains. Two configurations, and run both:
  - **Natural federation**: client = hospital, no Dirichlet anywhere. This is the configuration that
    answers "is the effect an artifact of synthetic partitions?" — it is the single most valuable
    robustness result in the paper.
  - **Sub-partitioned**: split each hospital into k clients so client count is comparable to D1–D3.
- **Subsample for the sweeps.** 455k patches × thousands of shadow federations is not affordable.
  Use a fixed, seeded, class- and hospital-stratified subsample (target ~60k patches) for sweeps, and
  the full set only for the headline non-shadow runs. Record the subsample indices in the manifest
  and report the subsampling in the paper. Do not quietly subsample differently per experiment.
- **Ethics.** Public research dataset, de-identified, used under its license. The attacks are run
  against *models*, and no re-identification of any individual is attempted or claimed. Say exactly
  this in the ethics statement.

## 6. D5/D6 — stretch datasets

- **FMoW-WILDS** (`wilds.get_dataset("fmow")`, ~100 GB): real timestamps 2002–2017 plus 5 geographic
  regions. It is the only dataset in the set with *genuine* temporal drift rather than a synthetic
  class split, so it directly addresses gap #1 of the FCL surveys. Expensive. Attempt only after P7
  lands, and use a temporal subsample (e.g. 3 years × 3 regions) if you attempt it at all.
- **DomainNet** (6 domains, 345 classes, ~48 GB) for standard domain-IL comparability. If disk or
  time is tight, **Office-Home** (15,500 images, 4 domains, ~0.7 GB) buys most of the comparability
  for 1.5% of the bytes. Prefer Office-Home if it is a choice between that and cutting D5.

---

## 7. Splits, and the one that is easy to get wrong

Four splits per dataset, disjoint, fixed by seed, recorded in the manifest:

| Split | Used by |
|---|---|
| `train` | the federation's clients |
| `test`  | accuracy / BWT evaluation |
| `ref`   | **the adversary's auxiliary public data** — A2's reference, A1's offline calibration |
| `canary`| inserted records for one-run auditing (A7); drawn from `ref`'s distribution, never from `train` |

`ref` must be **disjoint from `train`**. If the adversary's reference contains target records, every
attack number in the paper is inflated and the result is worthless. Add a test that asserts
emptiness of the intersection for every dataset, and make it one of the claim tests so it cannot
silently regress.

---

## 8. What each dataset is for, as a table for the paper

Produce this as `tables/tab00_datasets.csv` → `tables/tab00_datasets.tex` during P1, with real numbers
pulled from the manifest, not typed:

`dataset, n_images, n_classes, n_tasks_default, client_partition, natural_clients, image_size, license, role_in_paper`
