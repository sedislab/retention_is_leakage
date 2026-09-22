"""M6 — C²Prompt-style prompt pool (family F3 + F7), the current prompt SOTA per RESEARCH_PLAN.md's
method table. Unlike every other method in the zoo, this one needs backprop through the ViT and
cannot use the frozen feature cache — it operates on raw images via `run_prompt_method.py`, not
`sim.run` (that driver assumes a fixed feature matrix; this one needs a real forward/backward pass
through a partially-frozen network). `cacheable=False` is the load-bearing fact about this method.

**Simplifications versus the published method, stated rather than hidden** (CLAUDE.md non-negotiable
#8 — reimplement against the paper's description, record the gap, do not chase the original code):
- A single, fixed-size global prompt pool shared across the whole run (no per-task pool growth).
- Prompt tokens receive no positional embedding of their own — the backbone's pretrained positional
  embedding is applied only to [CLS, patch tokens] before prompts are spliced in after CLS, so the
  frozen backbone's pretrained positional structure is exactly preserved for the tokens it was
  trained on. This matches several public L2P/DualPrompt-family reimplementations' convention.
- Top-k prompt selection uses cosine similarity between a frozen-backbone query (the unprompted CLS
  token) and each pool entry's key — the standard L2P/DualPrompt query mechanism.

Trainable parameters: the prompt pool, the prompt keys, and the classifier head. Everything else in
the ViT (patch embedding, CLS token, positional embedding, transformer blocks, final norm) stays
frozen — `requires_grad=False`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..artifacts import ArtifactRecord, Family
from .base import MethodSpec


class PromptPoolViT(nn.Module):
    """Wraps a frozen timm ViT with a learnable, queryable prompt pool spliced in after CLS."""

    def __init__(self, backbone: nn.Module, embed_dim: int, n_classes: int, n_pool: int, prompt_length: int, top_k: int):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        self.n_pool = n_pool
        self.top_k = top_k
        self.prompt_length = prompt_length
        self.pool_prompts = nn.Parameter(torch.randn(n_pool, prompt_length, embed_dim) * 0.02)
        self.pool_keys = nn.Parameter(torch.randn(n_pool, embed_dim) * 0.02)
        self.head = nn.Linear(embed_dim, n_classes)

    def _query(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            tokens = self.backbone.patch_embed(x)
            tokens = self.backbone._pos_embed(tokens) if hasattr(self.backbone, "_pos_embed") else tokens
            feats = self.backbone.blocks(tokens)
            feats = self.backbone.norm(feats)
        return feats[:, 0]  # unprompted CLS token as the query

    def forward(self, x: torch.Tensor) -> tuple:
        """Returns `(logits, key_loss)`. `key_loss` is the negative mean cosine similarity between
        each input's query and its *selected* keys -- `.topk` returns non-differentiable indices
        (which keys get picked) alongside differentiable values (how similar the picked ones are);
        using only `.indices` (as an earlier version of this file did) silently drops the only
        gradient path into `pool_keys`, so the keys never train. Minimising `key_loss` pulls selected
        keys toward their queries, exactly the L2P/DualPrompt key-query alignment objective."""
        query = self._query(x)  # (B, D), no grad
        sim = F.normalize(query, dim=-1) @ F.normalize(self.pool_keys, dim=-1).T  # (B, n_pool)
        topk_sim, topk_idx = sim.topk(self.top_k, dim=-1)  # both (B, top_k)
        key_loss = -topk_sim.mean()
        selected = self.pool_prompts[topk_idx]  # (B, top_k, prompt_length, D)
        B = x.shape[0]
        selected = selected.reshape(B, self.top_k * self.prompt_length, -1)

        tokens = self.backbone.patch_embed(x)
        tokens = self.backbone._pos_embed(tokens) if hasattr(self.backbone, "_pos_embed") else tokens
        cls_tok, patch_tok = tokens[:, :1], tokens[:, 1:]
        spliced = torch.cat([cls_tok, selected, patch_tok], dim=1)

        feats = self.backbone.blocks(spliced)
        feats = self.backbone.norm(feats)
        cls_out = feats[:, 0]
        return self.head(cls_out), key_loss

    def trainable_parameters(self):
        return [self.pool_prompts, self.pool_keys, *self.head.parameters()]

    def state_snapshot(self) -> dict:
        return {
            "pool_prompts": self.pool_prompts.detach().clone(),
            "pool_keys": self.pool_keys.detach().clone(),
            "head_weight": self.head.weight.detach().clone(),
            "head_bias": self.head.bias.detach().clone(),
        }

    def load_snapshot(self, snap: dict) -> None:
        with torch.no_grad():
            self.pool_prompts.copy_(snap["pool_prompts"])
            self.pool_keys.copy_(snap["pool_keys"])
            self.head.weight.copy_(snap["head_weight"])
            self.head.bias.copy_(snap["head_bias"])


class PromptFCL:
    """Not an `FCLMethod` subclass -- see module docstring. Driven by `run_prompt_method.py`, which
    handles image loading, local training loops, and FedAvg; this class owns the model and the
    per-round artifact-emission logic so both stay in one place."""

    spec = MethodSpec(
        name="M6_c2prompt",
        families=(Family.PROMPT, Family.COUNTS),
        cacheable=False,
        retention_type="individual",
        retention_knob_name="n_pool",
    )

    def __init__(self, backbone: nn.Module, embed_dim: int, n_classes: int, config: dict, device: str):
        self.device = device
        self.n_pool = int(config.get("n_pool", 10))
        self.top_k = int(config.get("top_k", 4))
        self.prompt_length = int(config.get("prompt_length", 5))
        self.lr = float(config.get("lr", 0.01))
        self.local_epochs = int(config.get("local_epochs", 1))
        self.key_loss_weight = float(config.get("key_loss_weight", 0.5))
        self.model = PromptPoolViT(
            backbone, embed_dim, n_classes, self.n_pool, self.prompt_length, self.top_k
        ).to(device)
        self._round = 0

    def local_train(self, loader, snapshot_in: dict) -> tuple:
        """One client's local training from the given global snapshot. Returns (new_snapshot, n_seen,
        counts_per_class)."""
        self.model.load_snapshot(snapshot_in)
        opt = torch.optim.Adam(self.model.trainable_parameters(), lr=self.lr)
        n_seen = 0
        counts: dict = {}
        for _ in range(self.local_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                logits, key_loss = self.model(xb)
                loss = F.cross_entropy(logits, yb) + self.key_loss_weight * key_loss
                loss.backward()
                opt.step()
                n_seen += len(yb)
                for c in yb.tolist():
                    counts[c] = counts.get(c, 0) + 1
        return self.model.state_snapshot(), n_seen, counts

    @staticmethod
    def fedavg(snapshots: list, weights: list) -> dict:
        total = sum(weights)
        out = {}
        for key in snapshots[0]:
            out[key] = sum(w * s[key] for w, s in zip(weights, snapshots)) / total
        return out

    def make_records(self, round_idx: int, task_idx: int, client: int, touched_ids: frozenset,
                      snapshot: dict, counts: dict, n_classes: int) -> list:
        prompt_payload = {
            "pool_prompts": snapshot["pool_prompts"].cpu().numpy(),
            "pool_keys": snapshot["pool_keys"].cpu().numpy(),
        }
        counts_arr = np.zeros(n_classes, dtype=int)
        for c, n in counts.items():
            counts_arr[c] = n
        return [
            ArtifactRecord(
                round=round_idx, task=task_idx, client=client, family=Family.PROMPT,
                payload=prompt_payload, touched=touched_ids, n_touched=len(touched_ids),
                passes_over_data=self.local_epochs,
                meta={"n_pool": self.n_pool, "top_k": self.top_k, "prompt_length": self.prompt_length},
            ),
            ArtifactRecord(
                round=round_idx, task=task_idx, client=client, family=Family.COUNTS,
                payload={"counts": counts_arr}, touched=touched_ids, n_touched=len(touched_ids),
                passes_over_data=1, meta={},
            ),
        ]

    def predict(self, loader) -> tuple:
        self.model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                logits, _key_loss = self.model(xb)
                preds.append(logits.argmax(dim=-1).cpu().numpy())
                labels.append(yb.numpy())
        self.model.train()
        return np.concatenate(preds), np.concatenate(labels)
