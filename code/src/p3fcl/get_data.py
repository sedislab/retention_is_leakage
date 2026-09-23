"""Two-phase dataset fetch/prepare/audit per `build/02_DATASETS.md`.

`fetch(dataset)`  — network step: try the canonical URL then mirrors, record whichever worked plus
                    its sha256/bytes into `data/MANIFEST.json`. Safe on the login node or a batch job
                    (compute-node internet is confirmed, see `build/01_KODIAK.md §2`).
`prepare(dataset)`— no network. Extracts the raw archive, builds our own stratified train/test/ref
                    split (`canary` carved out of `ref`), and writes `datasets/<name>/index.json`:
                    `{n_images, n_classes, class_names, samples: [{id, path, label, ...}], splits,
                    natural_partition_field}`. `features.extract()` reads only this index.
`audit(dataset)`  — prints n_images/n_classes/class histogram/splits from the index. Never skip this.

`ref` must be disjoint from `train` (02_DATASETS.md §7) — every `_stratified_split` call below
partitions indices into disjoint named buckets by construction; `tests/test_splits.py` checks it
against the real manifest once one exists.
"""
from __future__ import annotations

import json
import pickle
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from . import rng as rng_mod
from .provenance import source_hash

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "raw"
DATASETS_DIR = REPO_ROOT / "datasets"
MANIFEST_PATH = REPO_ROOT / "data" / "MANIFEST.json"

SPLIT_SEED = 20260914  # fixed, so splits are reproducible and auditable — never change casually


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))


def _update_manifest(name: str, entry: dict) -> None:
    manifest = _load_manifest()
    existing = manifest.get(name, {})
    existing.update(entry)
    manifest[name] = existing
    _save_manifest(manifest)


def _download(urls: list, dest: Path) -> str:
    """Try each URL with `wget -c` into a `.tmp`, atomic-rename on success. Idempotent: returns
    immediately if `dest` already exists (a re-run costs nothing)."""
    if dest.exists():
        return "already-downloaded"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    last_err = None
    for url in urls:
        try:
            subprocess.run(
                ["wget", "-c", "--tries=3", "--timeout=60", "-O", str(tmp), url],
                check=True, capture_output=True, text=True,
            )
            if not tmp.exists() or tmp.stat().st_size == 0:
                raise RuntimeError("downloaded file is empty")
            tmp.replace(dest)
            return url
        except Exception as e:  # noqa: BLE001 — deliberately broad: fall through to the next mirror
            last_err = e
            if tmp.exists():
                tmp.unlink()
            continue
    raise RuntimeError(f"all URLs failed for {dest.name}: tried {urls}; last error: {last_err}")


def _stratified_split(labels: np.ndarray, seed: int, fracs: dict) -> dict:
    """Disjoint, stratified-per-class split of `range(len(labels))` into named buckets, e.g.
    `fracs={"train": 0.8, "test": 0.1, "ref": 0.1}`."""
    labels = np.asarray(labels)
    r = rng_mod.seeded("get_data._stratified_split", seed)
    names = list(fracs.keys())
    weights = np.array([fracs[n] for n in names], dtype=float)
    out = {name: [] for name in names}
    for c in sorted({int(x) for x in labels}):
        idx = np.where(labels == c)[0]
        idx = idx[r.permutation(len(idx))]
        cuts = (np.cumsum(weights) / weights.sum() * len(idx)).astype(int)
        starts = [0] + cuts[:-1].tolist()
        for name, s, e in zip(names, starts, cuts):
            out[name].extend(int(i) for i in idx[s:e])
    return {name: sorted(ids) for name, ids in out.items()}


def _carve_canary(ref_ids: list, seed: int, n: int = 500) -> list:
    """Canary pool for A7 one-run auditing: drawn from `ref`'s distribution, never from `train`."""
    if not ref_ids:
        return []
    r = rng_mod.seeded("get_data._carve_canary", seed)
    ref_ids = np.array(ref_ids)
    n = min(n, len(ref_ids))
    chosen = ref_ids[r.permutation(len(ref_ids))[:n]]
    return sorted(int(i) for i in chosen)


def _prepare_source_hash() -> str:
    return source_hash()


# --------------------------------------------------------------------------------------------- #
# D1 — CIFAR-100
# --------------------------------------------------------------------------------------------- #

CIFAR100_URLS = ["https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"]


def fetch_cifar100() -> dict:
    dest = RAW_DIR / "cifar-100-python.tar.gz"
    existing = _load_manifest().get("cifar100", {})
    url_used = _download(CIFAR100_URLS, dest)
    entry = {
        "url_used": existing.get("url_used") if url_used == "already-downloaded" else url_used,
        "mirrors_tried": CIFAR100_URLS,
        "sha256": _sha256(dest),
        "bytes": dest.stat().st_size,
        "license": "MIT-like research use (Krizhevsky, 2009)",
        "fetched_utc": existing.get("fetched_utc", _utc_now()),
    }
    _update_manifest("cifar100", entry)
    return entry


def prepare_cifar100() -> dict:
    archive = RAW_DIR / "cifar-100-python.tar.gz"
    if not archive.exists():
        raise FileNotFoundError(f"{archive} not found — run fetch_cifar100() first")

    out_root = DATASETS_DIR / "cifar100"
    extract_dir = out_root / "_raw_extract"
    root = extract_dir / "cifar-100-python"
    if not root.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive) as tf:
            tf.extractall(extract_dir)

    def _load_batch(name):
        with open(root / name, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        data = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.array(d[b"fine_labels"])
        return data, labels

    train_data, train_labels = _load_batch("train")
    test_data, test_labels = _load_batch("test")
    images = np.concatenate([train_data, test_data], axis=0)
    labels = np.concatenate([train_labels, test_labels], axis=0)

    with open(root / "meta", "rb") as f:
        meta = pickle.load(f, encoding="bytes")
    class_names = [c.decode("utf-8") for c in meta[b"fine_label_names"]]

    img_dir = out_root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for i in range(len(images)):
        rel = f"images/{i:06d}.png"
        p = out_root / rel
        if not p.exists():
            Image.fromarray(images[i]).save(p)
        samples.append({"id": i, "path": rel, "label": int(labels[i])})

    splits = _stratified_split(
        labels, seed=SPLIT_SEED, fracs={"train": 40000 / 60000, "test": 10000 / 60000, "ref": 10000 / 60000}
    )
    canary = _carve_canary(splits["ref"], seed=SPLIT_SEED)

    index = {
        "n_images": len(images), "n_classes": len(class_names), "class_names": class_names,
        "samples": samples, "splits": {**splits, "canary": canary},
        "natural_partition_field": None, "image_size": [32, 32],
    }
    (out_root / "index.json").write_text(json.dumps(index))

    entry = {
        "n_images": len(images), "n_classes": len(class_names), "natural_partition_field": None,
        "splits": {k: len(v) for k, v in splits.items()},
        "prepared_utc": _utc_now(), "prepare_source_hash": _prepare_source_hash(),
    }
    _update_manifest("cifar100", entry)
    return entry


# --------------------------------------------------------------------------------------------- #
# D2 — ImageNet-R
# --------------------------------------------------------------------------------------------- #

IMAGENET_R_URLS = [
    "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar",
    "https://huggingface.co/datasets/axiong/imagenet-r/resolve/main/imagenet-r.tar",
]


def fetch_imagenet_r() -> dict:
    dest = RAW_DIR / "imagenet-r.tar"
    existing = _load_manifest().get("imagenet_r", {})
    url_used = _download(IMAGENET_R_URLS, dest)
    entry = {
        "url_used": existing.get("url_used") if url_used == "already-downloaded" else url_used,
        "mirrors_tried": IMAGENET_R_URLS,
        "sha256": _sha256(dest),
        "bytes": dest.stat().st_size,
        "license": "Research use (Hendrycks et al., 2021)",
        "fetched_utc": existing.get("fetched_utc", _utc_now()),
    }
    _update_manifest("imagenet_r", entry)
    return entry


def prepare_imagenet_r() -> dict:
    archive = RAW_DIR / "imagenet-r.tar"
    if not archive.exists():
        raise FileNotFoundError(f"{archive} not found — run fetch_imagenet_r() first")

    out_root = DATASETS_DIR / "imagenet_r"
    extract_dir = out_root / "imagenet-r"
    if not extract_dir.exists():
        with tarfile.open(archive) as tf:
            tf.extractall(out_root)

    class_dirs = sorted(p for p in extract_dir.iterdir() if p.is_dir())
    class_names = [p.name for p in class_dirs]
    samples, labels = [], []
    sid = 0
    for label, cdir in enumerate(class_dirs):
        for img_path in sorted(cdir.glob("*.jpg")):
            samples.append({"id": sid, "path": str(img_path.relative_to(out_root)), "label": label})
            labels.append(label)
            sid += 1
    labels = np.array(labels)

    splits = _stratified_split(labels, seed=SPLIT_SEED, fracs={"train": 0.8, "test": 0.1, "ref": 0.1})
    canary = _carve_canary(splits["ref"], seed=SPLIT_SEED)

    index = {
        "n_images": len(samples), "n_classes": len(class_names), "class_names": class_names,
        "samples": samples, "splits": {**splits, "canary": canary}, "natural_partition_field": None,
    }
    (out_root / "index.json").write_text(json.dumps(index))

    entry = {
        "n_images": len(samples), "n_classes": len(class_names), "natural_partition_field": None,
        "splits": {k: len(v) for k, v in splits.items()},
        "prepared_utc": _utc_now(), "prepare_source_hash": _prepare_source_hash(),
        "caveat": "ViT-B/16 IN21k backbone has seen ImageNet-21k, which overlaps these classes — "
                  "state once in the paper, per 02_DATASETS.md §3.",
    }
    _update_manifest("imagenet_r", entry)
    return entry


# --------------------------------------------------------------------------------------------- #
# D3 — CUB-200-2011
# --------------------------------------------------------------------------------------------- #

CUB200_URLS = [
    "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz",
    "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz",
]


def fetch_cub200() -> dict:
    dest = RAW_DIR / "CUB_200_2011.tgz"
    existing = _load_manifest().get("cub200", {})
    url_used = _download(CUB200_URLS, dest)
    entry = {
        "url_used": existing.get("url_used") if url_used == "already-downloaded" else url_used,
        "mirrors_tried": CUB200_URLS,
        "sha256": _sha256(dest),
        "bytes": dest.stat().st_size,
        "license": "Research use (Caltech-UCSD Birds-200-2011)",
        "fetched_utc": existing.get("fetched_utc", _utc_now()),
    }
    _update_manifest("cub200", entry)
    return entry


def prepare_cub200() -> dict:
    archive = RAW_DIR / "CUB_200_2011.tgz"
    if not archive.exists():
        raise FileNotFoundError(f"{archive} not found — run fetch_cub200() first")

    out_root = DATASETS_DIR / "cub200"
    cub_dir = out_root / "CUB_200_2011"
    if not cub_dir.exists():
        with tarfile.open(archive) as tf:
            tf.extractall(out_root)

    def _read_lines(name):
        return (cub_dir / name).read_text().strip().splitlines()

    id_to_path = {}
    for line in _read_lines("images.txt"):
        iid, rel = line.split(maxsplit=1)
        id_to_path[int(iid)] = rel
    id_to_label = {}
    for line in _read_lines("image_class_labels.txt"):
        iid, cls = line.split()
        id_to_label[int(iid)] = int(cls) - 1
    id_to_official_train = {}
    for line in _read_lines("train_test_split.txt"):
        iid, is_train = line.split()
        id_to_official_train[int(iid)] = int(is_train)
    class_names = [line.split(maxsplit=1)[1] for line in _read_lines("classes.txt")]

    ids_sorted = sorted(id_to_path)
    samples, labels, official_train_flags = [], [], []
    for sid, orig_id in enumerate(ids_sorted):
        samples.append({"id": sid, "path": f"CUB_200_2011/images/{id_to_path[orig_id]}", "label": id_to_label[orig_id]})
        labels.append(id_to_label[orig_id])
        official_train_flags.append(id_to_official_train[orig_id])
    labels = np.array(labels)
    official_train_flags = np.array(official_train_flags)

    official_train_ids = np.where(official_train_flags == 1)[0]
    official_test_ids = np.where(official_train_flags == 0)[0]
    sub = _stratified_split(labels[official_train_ids], seed=SPLIT_SEED, fracs={"train": 0.85, "ref": 0.15})
    train_final = sorted(int(official_train_ids[i]) for i in sub["train"])
    ref_final = sorted(int(official_train_ids[i]) for i in sub["ref"])
    canary = _carve_canary(ref_final, seed=SPLIT_SEED, n=200)

    index = {
        "n_images": len(samples), "n_classes": len(class_names), "class_names": class_names,
        "samples": samples,
        "splits": {
            "train": train_final, "test": sorted(int(i) for i in official_test_ids),
            "ref": ref_final, "canary": canary,
        },
        "natural_partition_field": None,
    }
    (out_root / "index.json").write_text(json.dumps(index))

    entry = {
        "n_images": len(samples), "n_classes": len(class_names), "natural_partition_field": None,
        "splits": {"train": len(train_final), "test": len(official_test_ids), "ref": len(ref_final)},
        "prepared_utc": _utc_now(), "prepare_source_hash": _prepare_source_hash(),
    }
    _update_manifest("cub200", entry)
    return entry


# --------------------------------------------------------------------------------------------- #
# D4 — Camelyon17-WILDS
# --------------------------------------------------------------------------------------------- #

CAMELYON17_ROOT = DATASETS_DIR / "camelyon17_wilds_raw"


def fetch_camelyon17() -> dict:
    """`wilds.get_dataset(download=True)` downloads from `worksheets.codalab.org`, which times out on
    every connection attempt from the Kodiak login node (verified: TCP connect to 20.232.203.197:443
    times out after 10s; general outbound access is fine). Rather than let a caller sit on that hang,
    fail fast and point at the working path: `code/scripts/build_camelyon17_from_hf_mirror.py`, which
    fetches the same CC0 WILDS release from a Hugging Face community re-hosting
    (`wltjr1007/Camelyon17-WILDS`, verified identical schema/per-center counts) and reconstructs the
    on-disk layout `Camelyon17Dataset(download=False)` expects. See `data/MANIFEST.json`'s `camelyon17`
    entry and `notes/2026-09-21_fig18_camelyon17.md` for the full story."""
    raise RuntimeError(
        "fetch_camelyon17(): the official wilds.get_dataset(download=True) path is blocked from "
        "Kodiak (worksheets.codalab.org connection timeout) -- do not call this. Run "
        "`python code/scripts/build_camelyon17_from_hf_mirror.py` followed by "
        "`python code/scripts/build_camelyon17_from_hf_mirror.py --materialize-selected` instead; "
        "see data/MANIFEST.json's camelyon17 entry for details."
    )


def prepare_camelyon17(subsample_target: int = 60000) -> dict:
    """Class- and hospital-stratified subsample to ~`subsample_target` patches (02_DATASETS.md §5:
    455k patches x thousands of shadow federations is not affordable; the full set is reserved for
    the headline non-shadow runs). Loader is the `wilds` Dataset object itself, indexed by
    `wilds_index` — Camelyon17 patches are not materialised as separate files under `datasets/`.
    """
    from wilds import get_dataset

    ds = get_dataset(dataset="camelyon17", download=False, root_dir=str(CAMELYON17_ROOT))
    y = ds.y_array.numpy()
    metadata = ds.metadata_array.numpy()
    hospital_col = ds.metadata_fields.index("hospital")
    hospital = metadata[:, hospital_col]
    # Wave V3 (08_FIX_PLAN.md's H13 fix, "matched-5-client redesign"): `slide` (one of the 3 fields
    # `Camelyon17Dataset` actually exposes -- `metadata_fields == ['hospital', 'slide', 'y']`; the raw
    # per-image `patient`/`node` columns exist in the underlying metadata.csv but are not surfaced
    # through this API) is a genuine natural sub-unit within a hospital -- every patch from the same
    # physical microscopy slide is one indivisible group, unlike a random Dirichlet client split which
    # ignores this structure entirely. Recorded per sample so `streams.py` can partition a hospital's
    # task-data into clients BY SLIDE (never splitting one slide's patches across two clients) as the
    # "natural" arm of the redesigned FIG18 comparison, matched on client COUNT against a same-n_clients
    # Dirichlet arm -- isolating "real structure vs. random split" from "how many clients," which is
    # exactly the confound `agents/OPEN_QUESTIONS.md`'s H13 flagged in the original pilot (natural arm
    # had 1 client, dirichlet arm had 10).
    slide_col = ds.metadata_fields.index("slide")
    slide = metadata[:, slide_col]
    n_total = len(y)

    r = rng_mod.seeded("get_data.prepare_camelyon17", SPLIT_SEED)
    keep = []
    frac = min(1.0, subsample_target / n_total)
    for h in np.unique(hospital):
        for c in np.unique(y):
            idx = np.where((hospital == h) & (y == c))[0]
            k = max(1, int(round(len(idx) * frac))) if len(idx) else 0
            idx = idx[r.permutation(len(idx))[:k]]
            keep.extend(int(i) for i in idx)
    keep = sorted(set(keep))
    keep_labels = y[keep]

    local_splits = _stratified_split(keep_labels, seed=SPLIT_SEED, fracs={"train": 0.8, "test": 0.1, "ref": 0.1})
    splits = {name: sorted(keep[i] for i in ids) for name, ids in local_splits.items()}
    canary = _carve_canary(splits["ref"], seed=SPLIT_SEED)

    samples = [
        {"id": int(gid), "wilds_index": int(gid), "label": int(y[gid]), "domain": int(hospital[gid]),
         "slide": int(slide[gid])}
        for gid in keep
    ]

    out_root = DATASETS_DIR / "camelyon17"
    out_root.mkdir(parents=True, exist_ok=True)
    index = {
        "n_images": len(samples), "n_classes": 2, "class_names": ["normal", "tumor"],
        "samples": samples, "splits": {**splits, "canary": canary},
        "natural_partition_field": "domain",
        "natural_client_field": "slide",
        "loader": "wilds:camelyon17", "wilds_root_dir": str(CAMELYON17_ROOT),
        "subsample_of": int(n_total), "subsample_target": subsample_target,
    }
    (out_root / "index.json").write_text(json.dumps(index))

    entry = {
        "n_images_full": int(n_total), "n_images_subsampled": len(samples), "n_classes": 2,
        "natural_partition_field": "domain (hospital)",
        "splits": {k: len(v) for k, v in splits.items()},
        "subsample_note": f"class+hospital-stratified subsample to ~{subsample_target} of {n_total}",
        "prepared_utc": _utc_now(), "prepare_source_hash": _prepare_source_hash(),
    }
    _update_manifest("camelyon17", entry)
    return entry


# --------------------------------------------------------------------------------------------- #
# dispatch + audit
# --------------------------------------------------------------------------------------------- #

_FETCHERS = {
    "cifar100": fetch_cifar100,
    "imagenet_r": fetch_imagenet_r,
    "cub200": fetch_cub200,
    "camelyon17": fetch_camelyon17,
}
_PREPARERS = {
    "cifar100": prepare_cifar100,
    "imagenet_r": prepare_imagenet_r,
    "cub200": prepare_cub200,
    "camelyon17": prepare_camelyon17,
}


def fetch(dataset: str) -> dict:
    if dataset not in _FETCHERS:
        raise ValueError(f"unknown dataset {dataset!r}; known: {sorted(_FETCHERS)}")
    return _FETCHERS[dataset]()


def prepare(dataset: str) -> dict:
    if dataset not in _PREPARERS:
        raise ValueError(f"unknown dataset {dataset!r}; known: {sorted(_PREPARERS)}")
    return _PREPARERS[dataset]()


def audit(dataset: str) -> dict:
    index_path = DATASETS_DIR / dataset / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"no index.json for {dataset!r} — run prepare first")
    index = json.loads(index_path.read_text())
    labels = np.array([s["label"] for s in index["samples"]])
    counts = np.bincount(labels, minlength=index["n_classes"])
    report = {
        "dataset": dataset,
        "n_images": index["n_images"],
        "n_classes": index["n_classes"],
        "class_histogram_min": int(counts.min()),
        "class_histogram_median": float(np.median(counts)),
        "class_histogram_max": int(counts.max()),
        "splits": {k: len(v) for k, v in index["splits"].items()},
        "natural_partition_field": index.get("natural_partition_field"),
    }
    print(json.dumps(report, indent=2))
    return report
