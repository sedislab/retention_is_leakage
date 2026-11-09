"""R1 (`rebuttal/R1_PROTOCOL.md`): release-only membership scores for M2 (F6 Gaussian statistics) and
M3 (F5 task covariance / subspace).

M2 and M3 read a task's raw data only during that task. Their task-k release depends only on which of
task k's records are members, so it can be recomputed from the population mask without training a
federation. Every statistic below uses the same numpy expressions, in the same order, as
`TARGET.fit_task` / `FOT.fit_task`. The equivalence tests (`tests/test_rb_release_audit.py` and
`scripts/rb_r1_equivalence.py`) check it against the methods' own releases to 1e-10.
"""
from __future__ import annotations

import numpy as np

RELEASE_VIEWS = {"m2_target": ("broadcast", "client"), "m3_fot": ("broadcast", "server")}
M2_VAR_FLOOR = 1e-6       # `TARGET.fit_task`: `feats.var(axis=0) + 1e-6`, `np.maximum(..., 1e-6)`
M3_SUBSPACE_RANK = 8      # `shadow_runner._method_config("m3_fot")`, no override in V2
M3_SERVER_LAMBDA = 1.0    # GRAM-family payload ridge in the paper (M8 `ridge_lambda`); R1_PROTOCOL.md


# ---------------------------------------------------------------- M2 (TARGET-style Gaussian replay)

def m2_task_release(X: np.ndarray, y: np.ndarray, task_shards: list) -> tuple:
    """`(client_stats, generators)` for one task. `client_stats[(client, c)] = (mean, var)` is each
    client's F6 GENERATIVE payload; `generators[c] = (agg_mean, agg_var)` is the count-weighted
    broadcast generator `TARGET.fit_task` stores in `_generators`."""
    client_stats: dict = {}
    per_class_client_stats: dict = {}
    for shard in task_shards:
        idx = np.array(shard.ids, dtype=int)
        classes_here = sorted({int(c) for c in y[idx]})
        for c in classes_here:
            class_idx = idx[y[idx] == c]
            feats = X[class_idx]
            mean = feats.mean(axis=0)
            var = feats.var(axis=0) + M2_VAR_FLOOR
            per_class_client_stats.setdefault(c, []).append((mean, var, len(class_idx)))
            client_stats[(int(shard.client), c)] = (mean, var)
    generators = {}
    for c, stats in per_class_client_stats.items():
        counts = np.array([n for _, _, n in stats], dtype=float)
        means = np.stack([m for m, _, _ in stats])
        varss = np.stack([v for _, v, _ in stats])
        total = counts.sum()
        agg_mean = (counts[:, None] * means).sum(axis=0) / total
        agg_second_moment = (counts[:, None] * (varss + means**2)).sum(axis=0) / total
        agg_var = np.maximum(agg_second_moment - agg_mean**2, M2_VAR_FLOOR)
        generators[c] = (agg_mean, agg_var)
    return client_stats, generators


def diag_gauss_loglik(f: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    return float(np.sum(-0.5 * np.log(2.0 * np.pi * var) - 0.5 * (f - mean) ** 2 / var))


# ---------------------------------------------------------------- M3 (FOT subspace)

def m3_task_release(X: np.ndarray, task_shards: list, rank: int = M3_SUBSPACE_RANK) -> tuple:
    """`(R, V)`: the task's centred covariance (the GRAM record's `payload["R"]`) and its top-`rank`
    eigenvectors (`new_dirs`, the directions FOT adds to the projection subspace). `(None, None)`
    if the task has no members."""
    task_ids: set = set()
    for shard in task_shards:
        task_ids.update(int(i) for i in shard.ids)
    if not task_ids:
        return None, None
    Xt = X[sorted(task_ids)]
    Xc = Xt - Xt.mean(axis=0, keepdims=True)
    cov = Xc.T @ Xc
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    k = min(rank, X.shape[1])
    return cov, eigvecs[:, order[:k]]


def subspace_energy_fraction(G: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Row-wise `||V^T g||^2 / ||g||^2` for centred rows `g` of `G`."""
    num = np.sum((G @ V) ** 2, axis=1)
    den = np.sum(G**2, axis=1)
    return num / den


def ridge_leverage(G: np.ndarray, R: np.ndarray, lam: float = M3_SERVER_LAMBDA) -> np.ndarray:
    """Row-wise `g^T (R + lam I)^{-1} g`."""
    A = R + lam * np.eye(R.shape[0])
    sol = np.linalg.solve(A, G.T)
    return np.einsum("ij,ji->i", G, sol)


# ---------------------------------------------------------------- per-shadow scoring

def ref_task_centres(X_ref: np.ndarray, y_ref: np.ndarray, task_classes: list) -> list:
    """Mean public `ref` feature over each task's class set (centre for both M3 scores)."""
    return [X_ref[np.isin(y_ref, cls)].mean(axis=0) for cls in task_classes]


def release_scores(method: str, X: np.ndarray, y: np.ndarray, masked_stream: list, targets: list,
                   ref_centres=None) -> dict:
    """`{view: (n_targets, n_rounds)}`, NaN before each target's task k and constant from k on, since
    each release happens once at task k. NaN also where the needed statistic does not exist in this
    shadow (the PROTOTYPE `full` convention of `shadow_runner._score_target`)."""
    n_rounds = len(masked_stream)
    views = RELEASE_VIEWS[method]
    out = {v: np.full((len(targets), n_rounds), np.nan) for v in views}
    by_task: dict = {}
    for j, t in enumerate(targets):
        by_task.setdefault(int(t["task"]), []).append(j)

    for k, js in sorted(by_task.items()):
        shards = masked_stream[k]
        if method == "m2_target":
            client_stats, generators = m2_task_release(X, y, shards)
            for j in js:
                t = targets[j]
                f, c = X[t["target_id"]], int(y[t["target_id"]])
                if c in generators:
                    out["broadcast"][j, k:] = diag_gauss_loglik(f, *generators[c])
                cs = client_stats.get((int(t["client"]), c))
                if cs is not None:
                    out["client"][j, k:] = diag_gauss_loglik(f, *cs)
        elif method == "m3_fot":
            R, V = m3_task_release(X, shards)
            if R is None:
                continue
            G = X[[targets[j]["target_id"] for j in js]] - ref_centres[k][None, :]
            bc = subspace_energy_fraction(G, V)
            sv = ridge_leverage(G, R)
            for i, j in enumerate(js):
                out["broadcast"][j, k:] = bc[i]
                out["server"][j, k:] = sv[i]
        else:
            raise ValueError(f"release_scores: no release audit for {method!r}")
    return out


# ---------------------------------------------------------------- combined arm

def combine_log_lr(L_margin: np.ndarray, L_release: np.ndarray, once: bool) -> dict:
    """Combine per-round LiRA log-LRs, both `(n_shadows, n_targets, n_rounds)`. A missing release
    LLR contributes 0 (no information). Defined only where the margin LLR is defined.

    `once=False` (`view=combined`, the spec read literally): per-round sum, then the same
    trajectory cumsum as `run_lira.compute_log_lr_surfaces`. `once=True` (`combined_once`): the single
    release is counted once. trajectory = cumsum(L_margin) + L_b and last_round = L_margin + L_b,
    where L_b is the release LLR at the target's own task (constant from k on)."""
    Lb = np.where(np.isnan(L_release), 0.0, L_release)
    valid = ~np.isnan(L_margin)
    filled_m = np.where(valid, L_margin, 0.0)
    cum_valid = np.cumsum(valid, axis=2) > 0
    if once:
        last = np.where(valid, L_margin + Lb, np.nan)
        traj = np.where(cum_valid, np.cumsum(filled_m, axis=2) + Lb, np.nan)
    else:
        per_round = np.where(valid, L_margin + Lb, np.nan)
        filled = np.where(valid, per_round, 0.0)
        last = per_round
        traj = np.where(cum_valid, np.cumsum(filled, axis=2), np.nan)
    return {"log_lr": last, "trajectory": traj}
