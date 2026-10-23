"""Reproducible level-4 matches using the unchanged published referee search."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from colony_one import constants, mcts
from colony_one.env import ColonyOneEnv
from colony_one.onnx_runner import OnnxEvaluator
from metrics import local_glicko

ROOT = Path(__file__).resolve().parent


def wilson(wins, games):
    if not games:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = wins / games
    d = 1 + z * z / games
    centre = (p + z * z / (2 * games)) / d
    half = z * math.sqrt(p * (1 - p) / games + z * z / (4 * games * games)) / d
    return [centre - half, centre + half]


def summary(records):
    n = len(records)
    w = sum(x["a_won"] for x in records)
    by_side = {}
    for side in [constants.CYAN, constants.MAGENTA]:
        subset = [x for x in records if x["a_side"] == side]
        sw = sum(x["a_won"] for x in subset)
        by_side[side] = {
            "wins": sw,
            "games": len(subset),
            "wilson95": wilson(sw, len(subset)),
        }
    # Jeffreys smoothing prevents infinite differences on small perfect samples.
    p = (w + 0.5) / (n + 1)
    pairs = {}
    for record in records:
        pairs.setdefault(record["seed"], []).append(float(record["a_won"]))
    paired = [sum(v) / len(v) for v in pairs.values() if len(v) == 2]
    pair_ci = None
    if len(paired) > 1:
        prng = np.random.default_rng(91351)
        boot = prng.choice(paired, size=(10000, len(paired)), replace=True).mean(1)
        pair_ci = np.quantile(boot, [0.025, 0.975]).tolist()
    return {
        "wins": w,
        "losses": n - w,
        "games": n,
        "winrate": w / n if n else None,
        "wilson95": wilson(w, n),
        "by_side": by_side,
        "paired_bootstrap95": pair_ci,
        "elo_difference_smoothed": 400 * math.log10(p / (1 - p)),
        "local_glicko2": local_glicko(records),
        "mean_actions": sum(x["actions"] for x in records) / n if n else None,
        "note": "Local results only. Elo difference is descriptive, not server Glicko-2.",
    }


_EVALUATORS = {}


def game(task):
    model_a, model_b, seed, swap, simulations, scene, trace = task
    for path in [model_a, model_b]:
        if path not in _EVALUATORS:
            _EVALUATORS[path] = OnnxEvaluator(path)
    agents = [_EVALUATORS[model_a], _EVALUATORS[model_b]]
    rngs = [np.random.default_rng(seed), np.random.default_rng(seed + 1)]
    config = mcts.MCTSConfig(simulations=simulations)
    env = ColonyOneEnv(scene=np.asarray(json.loads(Path(scene).read_text()), np.float32))
    a_side = constants.MAGENTA if swap else constants.CYAN
    log = []
    start = time.perf_counter()
    while not env.terminated:
        i = 0 if env.current_team == a_side else 1
        action = mcts.search_action(
            env.board,
            env.current_team,
            env.actions_left,
            env.round_number,
            agents[i],
            config,
            rng=rngs[i],
        )
        if action is None:
            raise RuntimeError("Referee returned no legal action")
        if trace:
            log.append(
                {
                    "action": env.action,
                    "team": env.current_team,
                    "from": action.from_pos,
                    "to": action.to_pos,
                    "kind": action.kind,
                }
            )
        env.step(action)
    result = {
        "seed": seed,
        "a_side": a_side,
        "a_won": env.winner == a_side,
        "winner": env.winner,
        "actions": env.action - 1,
        "seconds": time.perf_counter() - start,
    }
    if trace:
        result["trace"] = log
        result["final_board"] = env.render()
    return result


def run(model_a, model_b, pairs, seed, workers, simulations=100, scene=None, trace=False):
    scene = str(scene or ROOT / "scenes/level4.json")
    tasks = [
        (
            str(Path(model_a).resolve()),
            str(Path(model_b).resolve()),
            seed + pair * 2,
            swap,
            simulations,
            scene,
            trace,
        )
        for pair in range(pairs)
        for swap in [False, True]
    ]
    results = []
    if workers == 1:
        iterator = map(game, tasks)
        for result in iterator:
            results.append(result)
            print(
                f"{len(results)}/{len(tasks)} A wins {sum(x['a_won'] for x in results)}",
                flush=True,
            )
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            for result in pool.imap(game, tasks):
                results.append(result)
                print(
                    f"{len(results)}/{len(tasks)} A wins "
                    f"{sum(x['a_won'] for x in results)}",
                    flush=True,
                )
    return {
        "model_a": str(model_a),
        "model_b": str(model_b),
        "sha256_a": hashlib.sha256(Path(model_a).read_bytes()).hexdigest(),
        "sha256_b": hashlib.sha256(Path(model_b).read_bytes()).hexdigest(),
        "scene_sha256": hashlib.sha256(Path(scene).read_bytes()).hexdigest(),
        "simulations": simulations,
        "seed": seed,
        "summary": summary(results),
        "games": results,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", required=True, type=Path)
    p.add_argument("--b", required=True, type=Path)
    p.add_argument("--pairs", type=int, default=10)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--simulations", type=int, default=100)
    p.add_argument("--scene", type=Path, default=ROOT / "scenes/level4.json")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    if args.pairs < 1 or args.workers < 1 or args.simulations < 1:
        p.error("pairs, workers and simulations must be positive")
    report = run(
        args.a,
        args.b,
        args.pairs,
        args.seed,
        args.workers,
        args.simulations,
        args.scene,
        args.trace,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
