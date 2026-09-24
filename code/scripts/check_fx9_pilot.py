#!/usr/bin/env python3
"""Check every pilot file, including all views and M4 global/aggregate equivalence."""

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    with (ROOT / "build/waves/v3_pilot.tsv").open() as f:
        rows = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    for r in rows:
        directory = Path(r["out_dir"])
        count = 0
        for i in range(int(r["start"]), int(r["start"]) + int(r["count"])):
            with np.load(directory / f"shadow_{i:06d}.npz") as z:
                for key in z.files:
                    _ = z[key]
                if r["method"] == "m4_proto":
                    np.testing.assert_allclose(
                        z["scores_global"], z["scores_aggregate"], atol=1e-10, rtol=0, equal_nan=True
                    )
                assert int(z["shadow_id"]) == i
                for view in z["views"]:
                    assert np.isfinite(z[f"scores_{view}"]).any(), view
            count += 1
        print(f'FX9-5 PILOT ACCEPT {r["method"]} imagenet_r loadable={count}/64', flush=True)


if __name__ == "__main__":
    main()
