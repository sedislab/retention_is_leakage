#!/usr/bin/env python3
"""Driver for M6 (family F3+F7, backprop-through-the-ViT methods that cannot use `sim.run`'s cached-
feature interface). Loads raw images directly from `datasets/<name>/index.json`, builds a federated
task stream over the same ids/labels convention as everything else, runs local prompt-pool training +
FedAvg per round, evaluates, and writes a small result JSON (same shape as
`run_utility_baseline.py`'s, so a future TAB05 refresh can absorb M6/M7 rows without a special case).

**This is a validation run, not a full sweep**: `--images_per_class_cap` subsamples each class so a
first real end-to-end GPU run is cheap and fast to debug, given how contested the `gpu` queue has
been all session. A full multi-seed, full-dataset M6 sweep is a separate, much larger GPU-hour
commitment intentionally not launched automatically — see `notes/` for the explicit scope decision.

Must run on a GPU node (`code/scripts/pbs/prompt_validation.pbs`), never the login node.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "src"))

from p3fcl import metrics as metrics_mod  # noqa: E402
from p3fcl import provenance, streams  # noqa: E402
from p3fcl.artifacts import Ledger  # noqa: E402
from p3fcl.methods.m6_prompt import PromptFCL  # noqa: E402

BACKBONE = "vit_base_patch16_224.augreg_in21k"


class ImageDataset(Dataset):
    def __init__(self, root: Path, samples: list, transform):
        self.root = root
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        img = Image.open(self.root / s["path"]).convert("RGB")
        return self.transform(img), s["label"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--n_tasks", type=int, default=2)
    ap.add_argument("--n_clients", type=int, default=2)
    ap.add_argument("--images_per_class_cap", type=int, default=20)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("run_prompt_method.py needs a GPU node -- got no CUDA device")
    device = "cuda"

    index = json.loads((REPO_ROOT / "datasets" / args.dataset / "index.json").read_text())
    samples = index["samples"]
    n_classes = index["n_classes"]
    labels = np.array([s["label"] for s in samples])
    train_ids = np.array(index["splits"]["train"])

    rng = np.random.default_rng(args.seed)
    keep = []
    for c in sorted(set(labels[train_ids].tolist())):
        idx_c = train_ids[labels[train_ids] == c]
        idx_c = idx_c[rng.permutation(len(idx_c))[: args.images_per_class_cap]]
        keep.extend(idx_c.tolist())
    keep = np.array(sorted(keep))

    stream = streams.build_stream(labels, keep, n_tasks=args.n_tasks, n_clients=args.n_clients, beta=0.5, seed=args.seed)

    backbone = timm.create_model(BACKBONE, pretrained=True, num_classes=0)
    embed_dim = getattr(backbone, "embed_dim", None) or backbone.num_features
    data_cfg = timm.data.resolve_data_config({}, model=backbone)
    transform = timm.data.create_transform(**data_cfg, is_training=False)

    method = PromptFCL(
        backbone, embed_dim, n_classes,
        config={"n_pool": 10, "top_k": 4, "prompt_length": 5, "lr": 0.01, "local_epochs": args.local_epochs},
        device=device,
    )

    root = REPO_ROOT / "datasets" / args.dataset
    global_snap = method.model.state_snapshot()
    ledger = Ledger()
    T = len(stream)
    acc_matrix = np.full((T, T), np.nan)
    task_eval_ids = [sorted({i for shard in stream[t] for i in shard.ids}) for t in range(T)]

    for t, shards in enumerate(stream):
        snapshots, weights = [], []
        for shard in shards:
            shard_samples = [samples[i] for i in shard.ids]
            loader = DataLoader(ImageDataset(root, shard_samples, transform), batch_size=args.batch_size, shuffle=True)
            new_snap, n_seen, counts = method.local_train(loader, global_snap)
            snapshots.append(new_snap)
            weights.append(n_seen)
            touched = frozenset(int(i) for i in shard.ids)
            ledger.extend(method.make_records(t, t, shard.client, touched, new_snap, counts, n_classes))
        global_snap = PromptFCL.fedavg(snapshots, weights)
        method.model.load_snapshot(global_snap)

        for k in range(t + 1):
            ids_k = task_eval_ids[k]
            if not ids_k:
                continue
            eval_samples = [samples[i] for i in ids_k]
            loader = DataLoader(ImageDataset(root, eval_samples, transform), batch_size=args.batch_size, shuffle=False)
            preds, labs = method.predict(loader)
            acc_matrix[t, k] = float((preds == labs).mean())
        print(f"task {t} done: acc row = {acc_matrix[t]}")

    result = {
        "method": "M6", "dataset": args.dataset, "n_tasks_requested": args.n_tasks, "n_tasks_actual": T,
        "n_clients": args.n_clients, "seed": args.seed, "images_per_class_cap": args.images_per_class_cap,
        "final_avg_acc": float(np.nanmean(acc_matrix[-1, :])),
        "bwt": metrics_mod.backward_transfer(acc_matrix),
        "avg_incremental_acc": metrics_mod.average_incremental_accuracy(acc_matrix),
        "n_ledger_records": len(ledger),
        "note": "SMALL VALIDATION RUN (images_per_class_cap subsample) -- not a full TAB05 sweep",
    }
    print(json.dumps(result, indent=2))

    out_dir = REPO_ROOT / "runs" / "prompt_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"M6_{args.dataset}_t{args.n_tasks}_s{args.seed}.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result))
    tmp.replace(out_path)

    config = {"seed": args.seed, "method": "M6", "dataset": args.dataset, "n_tasks": args.n_tasks,
              "images_per_class_cap": args.images_per_class_cap}
    manifest = provenance.run_manifest(config, seed=args.seed)
    provenance.finalize(manifest, [out_path])
    return 0


if __name__ == "__main__":
    sys.exit(main())
