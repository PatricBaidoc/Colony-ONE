"""Small, embedded residual learned from winning historical actions.

The residual does not replace rules, MCTS, or the tactical evaluator. It has
not been benchmarked and is deliberately bounded to modest policy changes.
"""

from pathlib import Path

import numpy as np


def add_learned_policy(g, weights_path, scale=0.2):
    """Return a dynamic-batch (B,64,64) residual; no learned value head."""
    with np.load(Path(weights_path), allow_pickle=False) as arrays:
        weights = {name: arrays[name].astype(np.float32) for name in arrays.files}
    flat = g.reshape("observation", [0, 1024])
    conditioned = g.op("Mul", flat, g.const(weights["input_scale"]))
    hidden = g.op(
        "Relu",
        g.add(
            g.op("MatMul", conditioned, g.const(weights["w1"])), g.const(weights["b1"])
        ),
    )
    scores = g.add(g.op("MatMul", hidden, g.const(weights["w2"])), g.const(weights["b2"]))
    source = g.op("Gather", scores, g.const(np.arange(64), dtype=np.int64), axis=1)
    target = g.op("Gather", scores, g.const(np.arange(64, 128), dtype=np.int64), axis=1)
    grid = g.add(g.reshape(source, [0, 64, 1]), g.reshape(target, [0, 1, 64]))
    # The bound keeps imperfect imitation from overwhelming an explicit
    # Kernel win, firewall deadline, or other tactical priority.
    residual = g.scale(g.op("Tanh", g.scale(grid, 0.5)), float(scale))
    return residual, None
