"""Train a compact NumPy policy residual from historical winner moves.

Protocol reconstruction supplies training examples only. This command runs
no test suite, tournament, held-out evaluation, or winrate measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

# Keep the small training job predictable on a CPU-only cloud instance.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

from colony_one import encoding
from colony_one.env import ColonyOneEnv

ROOT = Path(__file__).resolve().parent
MOVE = re.compile(
    r"^\s*(\d+)\.\s+(CYAN|MAGENTA)\s+\w+\s+([A-H][1-8])\s+->\s+([A-H][1-8])", re.M
)
WINNER = re.compile(r"\*\*\* (CYAN|MAGENTA) WINS")


def square(text):
    return 8 - int(text[1]), ord(text[0]) - ord("A")


def dataset(paths):
    scene = np.asarray(json.loads((ROOT / "scenes/level4.json").read_text()), np.float32)
    x, legal, chosen, sample_weights = [], [], [], []
    seen = set()
    info = {
        "records": 0,
        "accepted_protocols": 0,
        "excluded_missing_or_malformed": 0,
        "deduplicated_protocols": 0,
        "sources": [],
    }
    for path in paths:
        path = Path(path)
        info["sources"].append(
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
        if path.suffix == ".json":
            records = json.loads(path.read_text())
        else:
            records = [{"protocol": path.read_text(), "attached_demonstration": True}]
        for record in records:
            info["records"] += 1
            protocol = record.get("protocol") or ""
            winner = WINNER.findall(protocol)
            if not winner:
                info["excluded_missing_or_malformed"] += 1
                continue
            digest = hashlib.sha256(protocol.encode()).hexdigest()
            if digest in seen:
                info["deduplicated_protocols"] += 1
                continue
            seen.add(digest)
            env = ColonyOneEnv(scene=scene)
            staged = []
            last_loser = None
            try:
                matches = list(MOVE.finditer(protocol))
                if not matches:
                    raise ValueError("No moves")
                for move_index, match in enumerate(matches):
                    number, team, source, target = match.groups()
                    if int(number) != env.action or team != env.current_team:
                        raise ValueError("Unusable turn order")
                    fr, fc = encoding.canonical_pos(square(source), team)
                    tr, tc = encoding.canonical_pos(square(target), team)
                    action = (fr * 8 + fc) * 64 + tr * 8 + tc
                    indices = env.legal_action_indices()
                    if action not in indices:
                        raise ValueError("Unusable action")
                    if team == winner[-1]:
                        staged.append(
                            (
                                env.observation().reshape(-1).copy(),
                                indices.copy(),
                                action,
                                2.0 if record.get("attached_demonstration") else 1.0,
                            )
                        )
                    else:
                        last_loser = (
                            move_index,
                            env.observation().reshape(-1).copy(),
                            indices.copy(),
                            action,
                        )
                    env.step(action)
                # Incomplete protocols must not supply confidently labeled
                # examples; filtering is data preparation, not a test result.
                if not env.terminated or env.winner != winner[-1]:
                    raise ValueError("Incomplete or inconsistent winner label")
            except (ValueError, RuntimeError):
                info["excluded_missing_or_malformed"] += 1
                continue
            info["accepted_protocols"] += 1
            # Only the final losing action immediately before an explicit
            # one/two-action Kernel capture receives weak anti-imitation.
            # This is a credit-assignment heuristic, not a claim that a
            # saving alternative existed or that all losing moves were bad.
            if last_loser is not None:
                index, observation, indices, action = last_loser
                following = matches[index + 1 :]
                final_line = protocol[matches[-1].start() :].splitlines()[0]
                if (
                    1 <= len(following) <= 2
                    and len(indices) > 1
                    and all(move.group(2) == winner[-1] for move in following)
                    and "xKE" in final_line
                ):
                    staged.append((observation, indices, action, -0.15))
            for observation, indices, action, weight in staged:
                x.append(observation)
                legal.append(indices)
                chosen.append(action)
                sample_weights.append(weight)
    if not x:
        raise ValueError("No usable historical training examples")
    width = max(len(a) for a in legal)
    actions = np.zeros((len(x), width), np.int64)
    mask = np.zeros((len(x), width), bool)
    target = np.zeros(len(x), np.int64)
    for i, (indices, action) in enumerate(zip(legal, chosen, strict=True)):
        actions[i, : len(indices)] = indices
        mask[i, : len(indices)] = True
        target[i] = int(np.flatnonzero(indices == action)[0])
    info.update(
        training_positions=len(x),
        winner_action_positions=sum(weight > 0 for weight in sample_weights),
        weak_negative_positions=sum(weight < 0 for weight in sample_weights),
        max_legal_actions=width,
        labels=(
            "Winner actions plus weak final losing actions followed by <=2 "
            "winning actions ending explicit Kernel capture; no score-field interpretation"
        ),
    )
    return (
        np.asarray(x, np.float32),
        actions,
        mask,
        target,
        np.asarray(sample_weights, np.float32),
        info,
    )


def train(paths, output, epochs=30, seed=22872):
    start = time.perf_counter()
    x, actions, mask, target, importance, info = dataset(paths)
    rng = np.random.default_rng(seed)
    conditioning = np.ones((1, 1024), np.float32)
    conditioning[:, 13 * 64 :] = 0.125
    x = x * conditioning
    w = {
        "w1": rng.normal(0, 0.035, (1024, 32)).astype(np.float32),
        "b1": np.zeros(32, np.float32),
        "w2": rng.normal(0, 0.035, (32, 128)).astype(np.float32),
        "b2": np.zeros(128, np.float32),
    }
    m = {k: np.zeros_like(v) for k, v in w.items()}
    v = {k: np.zeros_like(a) for k, a in w.items()}
    history = []
    step = 0
    for epoch in range(epochs):
        losses = []
        for batch_start in range(0, len(x), 64):
            if batch_start == 0:
                order = rng.permutation(len(x))
            ix = order[batch_start : batch_start + 64]
            xx = x[ix]
            aa = actions[ix]
            valid = mask[ix]
            yy = target[ix]
            n = len(ix)
            rows = np.arange(n)[:, None]
            z = xx @ w["w1"] + w["b1"]
            h = np.maximum(z, 0)
            scores = h @ w["w2"] + w["b2"]
            source = aa // 64
            destination = 64 + aa % 64
            logits = scores[rows, source] + scores[rows, destination]
            logits = np.where(valid, logits, -1e9)
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            bad = importance[ix] < 0
            weight = np.abs(importance[ix])
            weight = weight / weight.sum()
            selected_probability = probabilities[np.arange(n), yy].copy()
            positive_loss = -np.log(np.maximum(selected_probability, 1e-9))
            negative_loss = -np.log(np.maximum(1 - selected_probability, 1e-6))
            losses.append(
                float(np.sum(np.where(bad, negative_loss, positive_loss) * weight))
            )
            gradient = probabilities.copy()
            gradient[np.arange(n), yy] -= 1
            gradient[bad] *= (
                -selected_probability[bad]
                / np.maximum(1 - selected_probability[bad], 1e-6)
            )[:, None]
            gradient *= weight[:, None]
            ds = np.zeros((n, 128), np.float32)
            np.add.at(ds, (np.broadcast_to(rows, source.shape), source), gradient)
            np.add.at(
                ds, (np.broadcast_to(rows, destination.shape), destination), gradient
            )
            dh = (ds @ w["w2"].T) * (z > 0)
            grads = {
                "w2": h.T @ ds + 1e-4 * w["w2"],
                "b2": ds.sum(axis=0),
                "w1": xx.T @ dh + 1e-4 * w["w1"],
                "b1": dh.sum(axis=0),
            }
            step += 1
            for key, g in grads.items():
                g = np.clip(g, -2, 2)
                m[key] = 0.9 * m[key] + 0.1 * g
                v[key] = 0.999 * v[key] + 0.001 * g * g
                w[key] -= (
                    0.003
                    * (m[key] / (1 - 0.9**step))
                    / (np.sqrt(v[key] / (1 - 0.999**step)) + 1e-8)
                )
        history.append(float(np.mean(losses)))
    output = Path(output)
    np.savez_compressed(output, **w, input_scale=conditioning)
    info.update(
        seed=seed,
        epochs=epochs,
        architecture="1024->32 ReLU->128 source/target scores",
        training_loss=history,
        elapsed_seconds=time.perf_counter() - start,
        artifact=str(output),
        artifact_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        evaluation="Not performed. Training loss is not winrate or held-out performance.",
        method=(
            "Winner-action imitation plus weight-0.15 final-error anti-imitation; "
            "not deep-search teacher or AlphaZero"
        ),
    )
    (ROOT / "results/learning_summary.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output", type=Path, default=ROOT / "learned_weights.npz")
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.paths or [
        ROOT / "results/provided_matches.json",
        ROOT / "results/new_tournament_matches.json",
        ROOT / "results/match-22872.txt",
    ]
    train(paths, args.output, args.epochs)
