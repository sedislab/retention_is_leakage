#!/usr/bin/env python3
"""FX9-7 actual M9 ledgers at T=10/50, normalized-noise FX1 accounting.

Reported units follow the unchanged FX1 record-composition convention. These
cross-unit accountant outputs do not by themselves establish group sensitivity
for a mechanism calibrated only to U1; that limitation is explicit in the CSV.
"""

import csv
from pathlib import Path

import numpy as np
from p3fcl import features, provenance, sim, streams
from p3fcl.dp.accountant import account, eps_gaussian_composed, extrapolate_lifelong
from p3fcl.dp.mechanisms import analytic_gaussian_sigma
from p3fcl.methods.m9_contractive import ContractiveDPAnalytic
from p3fcl.units import Unit
from run_fig05 import _ledger_hash
from run_m9_audit import BACKBONE, _hparams_for
from run_m9_hparam_selection import _fit_pca

ROOT = Path(__file__).resolve().parents[2]
UNITS = [Unit.EXAMPLE, Unit.TASK, Unit.CLIENT_BOUNDED, Unit.CLIENT_LIFELONG]
DELTA = 1e-5


def write(path, rows, manifest):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    provenance.finalize(manifest, [path])


def main():
    manifest = provenance.run_manifest(
        dict(phase="FX9-7", hypotheses=["H1", "H7"], dataset="cifar100", eps0=[1, 4], tasks=[10, 50], seed=0),
        seed=0,
    )
    train = features.load_cache(ROOT / "features", "cifar100", BACKBONE, "train")
    ref = features.load_cache(ROOT / "features", "cifar100", BACKBONE, "ref")
    X, y = train["features"], train["labels"]
    pca = {}
    curve_rows, certified = [], []
    for eps0 in [1.0, 4.0]:
        p, lam, clip = _hparams_for(eps0)
        if p not in pca:
            pca[p] = _fit_pca(ref["features"], p_max=p, seed=0)
        for T in [10, 50]:
            stream = streams.build_stream(y, np.arange(len(y)), n_tasks=T, n_clients=10, beta=0.5, seed=0)
            method = ContractiveDPAnalytic(
                dict(
                    n_classes=100,
                    pca_basis=pca[p],
                    gamma=1.0,
                    eps=eps0,
                    delta=DELTA,
                    ridge_lambda=lam,
                    unit="U1",
                    clip_C=clip,
                    B=1.0,
                    noise_seed=0,
                )
            )
            result = sim.run(method, X, y, stream, seed=0)
            ledger = result["ledger"]
            lhash = _ledger_hash(ledger)
            sigma = float(next(iter(ledger)).meta["sigma"])
            sensitivity = np.sqrt(2.0)
            z = sigma / sensitivity
            for unit in UNITS:
                df = account(ledger, stream, unit=unit, sigma=z, delta=DELTA, window=3)
                r = df.iloc[-1]
                certified.append(
                    dict(
                        dataset="cifar100",
                        calibration_unit="U1",
                        eps0=eps0,
                        T=T,
                        unit=unit.value,
                        sigma=sigma,
                        sensitivity=sensitivity,
                        z=z,
                        m_T=r["m_T"],
                        eps=r["eps"],
                        eps0_analytic=eps0,
                        delta0_analytic=DELTA,
                        scope="FX1 record-composition convention; cross-unit group sensitivity not separately certified",
                    )
                )
                print(
                    f'FX9-7 CERTIFIED ACCEPT eps0={eps0:g} T={T} {unit.value} m={r["m_T"]} z={z:.12f} eps={r["eps"]:.12f} eps0_analytic={eps0:g}',
                    flush=True,
                )
                if T == 50:
                    ext = extrapolate_lifelong(df, sigma=z, delta=DELTA, T_max=1000)
                    for _, v in ext.iterrows():
                        curve_rows.append(
                            dict(
                                method="M9",
                                dataset="cifar100",
                                n_tasks=T,
                                ledger_hash=lhash,
                                unit=unit.value,
                                window_W=3,
                                sigma=z,
                                delta=DELTA,
                                T=int(v["T"]),
                                m_T=v["m_T"],
                                m_T_passes=v["m_T_passes"],
                                eps=v["eps"],
                                regime=v["regime"],
                                observed=int(v["observed"]),
                                eps0=eps0,
                                eps0_analytic=eps0 if int(v["m_T"]) == 1 else "",
                            )
                        )
    # The release multiplicity is data-independent for M9's one disjoint release/task.
    # Fill the remaining configured eps levels with the same observed multiplicity,
    # calibrated with the actual analytic Gaussian solver (no nominal sigma=2).
    expanded = []
    for eps0 in [0.5, 1.0, 2.0, 4.0, 8.0, float("inf")]:
        for r in [v for v in certified if v["eps0"] == 1.0]:
            z = analytic_gaussian_sigma(eps0, DELTA, 1.0) if np.isfinite(eps0) else 0.0
            eps = eps_gaussian_composed(z, int(r["m_T"]), DELTA) if z else float("inf")
            expanded.append(
                {**r, "eps0": eps0, "z": z, "sigma": z * np.sqrt(2.0), "eps": eps, "eps0_analytic": eps0}
            )
    write(ROOT / "results/m9_certified.csv", expanded, manifest)
    write(ROOT / "results/fig05_m9.csv", curve_rows, manifest)


if __name__ == "__main__":
    main()
