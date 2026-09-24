"""Only registry of shadow-store roots. Existence on disk never selects a generation."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_SHADOW_ROOT = Path("shadows")
V2_SHADOW_ROOT = Path("shadows_v2")
V3_SHADOW_ROOT = Path("shadows_v3")
SHADOW_ROOT = {
    "m0_fedavg": LEGACY_SHADOW_ROOT,
    "m1_glfc": V3_SHADOW_ROOT,
    "m2_target": V3_SHADOW_ROOT,
    "m3_fot": V2_SHADOW_ROOT,
    "m4_proto": V3_SHADOW_ROOT,
    "m5_hybrid_replay": V3_SHADOW_ROOT,
    "m8_analytic": V2_SHADOW_ROOT,
    "m9_contractive": V2_SHADOW_ROOT,
}


def shadow_dir(dataset, method, seed=0, root=REPO_ROOT):
    path = Path(root) / SHADOW_ROOT[method] / dataset / method
    return path if method == "m0_fedavg" and seed == 0 else path / f"seed{seed}"


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variant")
    p.add_argument("--legacy", action="store_true")
    p.add_argument("--v2", action="store_true")
    args = p.parse_args()
    if args.variant or args.legacy or args.v2:
        base = LEGACY_SHADOW_ROOT if args.legacy else V2_SHADOW_ROOT if args.v2 else SHADOW_ROOT[args.method]
        path = REPO_ROOT / base / args.dataset / args.method
        if args.variant:
            path /= args.variant
        if args.variant or not args.legacy or args.seed:
            path /= f"seed{args.seed}"
    else:
        path = shadow_dir(args.dataset, args.method, args.seed)
    print(path)


if __name__ == "__main__":
    main()
