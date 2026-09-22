"""Frozen-backbone feature cache. `extract()` needs torch/timm + a GPU node and is deliberately
unimplemented until Phase P1 — never call it from the login node.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def cache_key(dataset: str, backbone: str, split: str) -> str:
    blob = f"{dataset}|{backbone}|{split}".encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def save_cache(out_dir, dataset: str, backbone: str, split: str, features, cls_tokens, labels, ids, meta=None) -> Path:
    """Writes `<key>.npz` (atomically) + a sidecar `<key>.json` with n, d, dataset, backbone, split."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(dataset, backbone, split)
    npz_path = out_dir / f"{key}.npz"
    tmp_path = out_dir / f"{key}.npz.tmp"
    features = np.asarray(features)
    with open(tmp_path, "wb") as f:
        np.savez_compressed(
            f,
            features=features,
            cls_tokens=np.asarray(cls_tokens),
            labels=np.asarray(labels),
            ids=np.asarray(ids),
        )
    tmp_path.replace(npz_path)
    side = {
        "dataset": dataset,
        "backbone": backbone,
        "split": split,
        "n": int(features.shape[0]),
        "d": int(features.shape[1]) if features.ndim > 1 else None,
        "key": key,
    }
    if meta:
        side.update(meta)
    (out_dir / f"{key}.json").write_text(json.dumps(side, indent=2))
    return npz_path


def load_cache(out_dir, dataset: str, backbone: str, split: str) -> dict:
    out_dir = Path(out_dir)
    key = cache_key(dataset, backbone, split)
    npz_path = out_dir / f"{key}.npz"
    side_path = out_dir / f"{key}.json"
    if not npz_path.exists():
        raise FileNotFoundError(f"no feature cache for {dataset}|{backbone}|{split} at {npz_path}")
    with open(npz_path, "rb") as f:
        data = np.load(f)
        out = {
            "features": data["features"],
            "cls_tokens": data["cls_tokens"],
            "labels": data["labels"],
            "ids": data["ids"],
        }
    out["meta"] = json.loads(side_path.read_text()) if side_path.exists() else {}
    return out


def extract(
    dataset: str,
    backbone: str,
    split: str,
    batch_size: int = 128,
    device: str = "cuda",
    amp: bool = True,
    num_workers: int = 4,
    out_dir=None,
) -> Path:
    """timm, `pretrained=True, num_classes=0`, `torch.no_grad()`, AMP. Stores the pooled feature AND
    the pre-norm CLS token, unnormalised — A2 (Gram inversion) needs the exact vector the method
    would consume; L2 normalisation is a *method* decision, not an extraction-time one.

    Reads `datasets/<dataset>/index.json` (written by `get_data.prepare`). Camelyon17's index has
    `"loader": "wilds:camelyon17"` and no per-sample file path — its images are loaded through the
    `wilds` Dataset object by `wilds_index` instead of `PIL.Image.open`.

    Kodiak policy: never call this from the login node — submit `extract_features.pbs`.
    """
    import json as _json

    import numpy as _np
    import timm
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset as TorchDataset

    from .get_data import DATASETS_DIR
    from .get_data import REPO_ROOT as _REPO_ROOT

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "extract(device='cuda') but torch.cuda.is_available() is False — "
            "are you accidentally running this on the login node?"
        )

    index_path = DATASETS_DIR / dataset / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"no index.json for {dataset!r} at {index_path} — run get_data.prepare first")
    index = _json.loads(index_path.read_text())
    if split not in index["splits"]:
        raise ValueError(f"split {split!r} not in {dataset!r}'s splits: {sorted(index['splits'])}")
    split_ids = set(index["splits"][split])
    samples = [s for s in index["samples"] if s["id"] in split_ids]
    if not samples:
        raise ValueError(f"split {split!r} of {dataset!r} is empty")

    loader_kind = index.get("loader", "file")
    dataset_root = DATASETS_DIR / dataset

    model = timm.create_model(backbone, pretrained=True, num_classes=0)
    model.eval().to(device)
    data_cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**data_cfg, is_training=False)

    _wilds_ds_holder: dict = {}

    def _wilds_ds():
        if "ds" not in _wilds_ds_holder:
            from wilds import get_dataset

            _wilds_ds_holder["ds"] = get_dataset(
                dataset="camelyon17", download=False, root_dir=index["wilds_root_dir"]
            )
        return _wilds_ds_holder["ds"]

    class _SampleDataset(TorchDataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, i):
            s = samples[i]
            if loader_kind == "wilds:camelyon17":
                img = _wilds_ds().get_input(s["wilds_index"])
            else:
                img = Image.open(dataset_root / s["path"]).convert("RGB")
            return transform(img), s["label"], s["id"]

    dataloader = DataLoader(_SampleDataset(), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Capture the CLS token right before the final LayerNorm (timm ViTs expose it as `model.norm`),
    # via a forward pre-hook — robust to timm's internal `forward_features` doing the norm itself.
    captured: dict = {}
    hook_handle = None
    if hasattr(model, "norm"):
        def _pre_norm_hook(_module, args):
            captured["pre_norm_tokens"] = args[0].detach()

        hook_handle = model.norm.register_forward_pre_hook(_pre_norm_hook)

    all_pooled, all_cls, all_labels, all_ids = [], [], [], []
    use_amp = amp and device == "cuda"
    try:
        with torch.no_grad():
            for imgs, labels, ids in dataloader:
                imgs = imgs.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    feats = model.forward_features(imgs)
                    pooled = model.forward_head(feats, pre_logits=True)
                if "pre_norm_tokens" in captured:
                    cls_tok = captured.pop("pre_norm_tokens")[:, 0]
                elif feats.ndim == 3:
                    cls_tok = feats[:, 0]
                else:
                    cls_tok = pooled
                all_pooled.append(pooled.float().cpu().numpy())
                all_cls.append(cls_tok.float().cpu().numpy())
                all_labels.append(labels.numpy())
                all_ids.append(ids.numpy())
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    features_arr = _np.concatenate(all_pooled, axis=0)
    cls_arr = _np.concatenate(all_cls, axis=0)
    labels_arr = _np.concatenate(all_labels, axis=0)
    ids_arr = _np.concatenate(all_ids, axis=0)

    out_dir = Path(out_dir) if out_dir else (_REPO_ROOT / "features")
    return save_cache(out_dir, dataset, backbone, split, features_arr, cls_arr, labels_arr, ids_arr)
