"""Small, reproducible parameter tournament against model11 (tuning seeds only)."""

import argparse
import json

from arena import ROOT, run
from build_model import build

ORIGINAL = {
    "capture_scale": 1.0,
    "central_scale": 1.0,
    "material_scale": 1.0,
    "unsafe": 0.5,
    "guard_bonus": 0.06,
    "burn": 0.65,
    "attack": 0.35,
    "exact": False,
}
VARIANTS = {
    "exact_only": {**ORIGINAL, "exact": True},
    "soft": {},
    "tempo": {"tempo": 0.3, "risk": 0.5, "policy_risk": 1.0, "kernel_policy_risk": 1.0},
    "aggressive": {
        "unsafe": 0.1,
        "material_scale": 2.0,
        "attack": 0.5,
        "tempo": 0.45,
        "risk": 0.5,
        "policy_risk": 0.75,
        "kernel_policy_risk": 1.5,
    },
    "safety": {
        "unsafe": 0.5,
        "risk": 0.75,
        "policy_risk": 2.0,
        "kernel_policy_risk": 2.0,
        "tempo": 0.15,
    },
    "capture": {
        "capture_scale": 1.1,
        "material_scale": 2.0,
        "attack": 0.6,
        "tempo": 0.45,
    },
    "control": {"risk": 0.5, "pressure": 0.03, "tempo": 0.25, "policy_risk": 1.0},
    "sharp": {
        "capture_scale": 0.5,
        "unsafe": 0.0,
        "risk": 1.0,
        "material_scale": 1.5,
        "attack": 0.7,
        "tempo": 0.65,
        "policy_risk": 1.25,
        "kernel_policy_risk": 1.5,
    },
    "retune": {
        **ORIGINAL,
        "capture_scale": 0.6,
        "unsafe": 0.2,
        "material_scale": 1.5,
        "attack": 0.5,
    },
    "forced_tempo": {
        "tempo": 0.3,
        "risk": 0.5,
        "policy_risk": 1.0,
        "kernel_policy_risk": 1.0,
        "force_win": True,
        "force_policy": 4.0,
    },
    "forced_sharp": {
        "capture_scale": 0.5,
        "unsafe": 0.0,
        "risk": 1.0,
        "material_scale": 1.5,
        "attack": 0.7,
        "tempo": 0.65,
        "policy_risk": 1.25,
        "kernel_policy_risk": 1.5,
        "force_win": True,
        "force_policy": 4.0,
    },
    "forced_base": {**ORIGINAL, "exact": True, "force_win": True, "force_policy": 4.0},
    "two_guards": {
        **ORIGINAL,
        "exact": True,
        "force_win": True,
        "force_policy": 4.0,
        "unsafe": 0.25,
        "guard_bonus": 0.3,
        "guard_policy": True,
    },
    "guards_tempo": {
        "tempo": 0.3,
        "risk": 0.5,
        "policy_risk": 1.0,
        "kernel_policy_risk": 1.0,
        "unsafe": 0.25,
        "guard_bonus": 0.25,
        "guard_policy": True,
        "force_win": True,
        "force_policy": 4.0,
    },
    "trace": {
        **ORIGINAL,
        "exact": True,
        "force_win": True,
        "force_policy": 4.0,
        "piece_scales": [1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
        "guard_policy": True,
    },
    "shooters": {
        **ORIGINAL,
        "exact": True,
        "force_win": True,
        "force_policy": 4.0,
        "piece_scales": [1.0, 1.6, 0.9, 1.8, 1.2, 1.0],
        "guard_policy": True,
    },
    "force_soft": {
        **ORIGINAL,
        "exact": True,
        "force_win": True,
        "force_policy": 4.0,
        "capture_scale": 0.7,
        "guard_policy": True,
    },
    "follow_base": {
        **ORIGINAL,
        "exact": True,
        "force_win": True,
        "force_policy": 4.0,
        "followup": 0.35,
    },
    "follow_trace": {
        **ORIGINAL,
        "exact": True,
        "force_win": True,
        "force_policy": 4.0,
        "piece_scales": [1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
        "guard_policy": True,
        "followup": 0.35,
    },
    "follow_tempo": {
        "tempo": 0.3,
        "risk": 0.5,
        "policy_risk": 1.0,
        "kernel_policy_risk": 1.0,
        "force_win": True,
        "force_policy": 4.0,
        "followup": 0.35,
    },
    "follow_hard": {
        **ORIGINAL,
        "exact": True,
        "force_win": True,
        "force_policy": 4.0,
        "piece_scales": [1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
        "guard_policy": True,
        "followup": 0.7,
    },
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=int, default=6)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--only", nargs="*")
    p.add_argument("--tag", default="screen")
    args = p.parse_args()
    all_results = []
    for name, cfg in VARIANTS.items():
        if args.only and name not in args.only:
            continue
        path = ROOT / f"candidates/{name}.onnx"
        actual = build(path, cfg)
        path.with_suffix(".json").write_text(json.dumps(actual, indent=2) + "\n")
        print(f"\nCANDIDATE {name}", flush=True)
        result = run(
            path, ROOT / "baselines/model11.onnx", args.pairs, args.seed, args.workers
        )
        (ROOT / f"results/{args.tag}_{name}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        all_results.append({"name": name, "config": actual, **result["summary"]})
        print(json.dumps(all_results[-1]), flush=True)
        (ROOT / f"results/{args.tag}_summary.json").write_text(
            json.dumps(all_results, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
