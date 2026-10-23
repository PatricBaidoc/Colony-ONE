"""Bounded tactical priors; all game transitions remain in the referee MCTS.

The attached match 22872 is won by Magenta through coordinated exchanges,
two-action approaches and finally Ping C3-D3-E2 against the bare Kernel.
These features transfer the tactical ideas without hardcoding that line.
"""

from __future__ import annotations

import numpy as np
import onnx


def advanced_tactics(
    g,
    p,
    ownocc,
    enemyocc,
    attacks,
    clear,
    relocate,
    masks,
    shots,
    force_target,
    strength=1.0,
):
    policies, values = [], []
    near = np.zeros((64, 64), np.float32)
    depth = np.asarray(
        [min(i // 8, i % 8, 7 - i // 8, 7 - i % 8) for i in range(64)], np.float32
    )
    for f in range(64):
        for t in range(64):
            near[f, t] = max(abs(f // 8 - t // 8), abs(f % 8 - t % 8)) <= 2
    enemy_near = g.op("MatMul", p[11], g.const(near))
    own_nonking = g.op("Sub", ownocc, p[5])
    enemy_nonking = g.op("Sub", enemyocc, p[11])
    own_count = g.reshape(g.reduce(own_nonking, [1, 2]), [0, 1, 1])
    enemy_count = g.reshape(g.reduce(enemy_nonking, [1, 2]), [0, 1, 1])
    own_bare = g.op("Less", own_count, g.const(0.5))
    enemy_last = g.op("Equal", enemy_count, g.const(1.0))
    last_action = g.op("Less", p[13], g.const(0.75))
    endgame = g.op("Cast", g.op("And", own_bare, last_action), to=onnx.TensorProto.FLOAT)
    # Only an own Kernel can act when our non-Kernel count is zero. A
    # guarded enemy-Kernel attack kills the last guard without relocating.
    guarded_enemy = g.reshape("enemy_guarded_marker", [0, 1, 64])
    victims = g.add(enemy_nonking, guarded_enemy)
    capture_last = g.op(
        "Mul", victims, g.op("Cast", enemy_last, to=onnx.TensorProto.FLOAT)
    )
    already_bare = g.op(
        "Cast", g.op("Less", enemy_count, g.const(0.5)), to=onnx.TensorProto.FLOAT
    )
    triggers = g.add(capture_last, already_bare)
    # Capturing an unprotected enemy Kernel wins immediately and takes
    # precedence over every end-of-turn rule.
    excludes_win = g.op("Sub", g.const(1.0), g.op("Sub", p[11], guarded_enemy))
    trigger = g.op(
        "Mul",
        g.op("Mul", g.reshape(p[5], [0, 64, 1]), triggers),
        g.op("Mul", endgame, excludes_win),
    )
    distance_after = g.add(
        g.op("Mul", relocate, enemy_near),
        g.op(
            "Mul", g.op("Sub", g.const(1.0), relocate), g.reshape(enemy_near, [0, 64, 1])
        ),
    )
    destination_depth = g.const(depth.reshape(1, 1, 64))
    origin_depth = g.const(depth.reshape(1, 64, 1))
    own_depth = g.add(
        g.op("Mul", relocate, destination_depth),
        g.op("Mul", g.op("Sub", g.const(1.0), relocate), origin_depth),
    )
    enemy_depth = g.reshape(
        g.reduce(g.op("Mul", p[11], g.const(depth.reshape(1, 1, 64))), [1, 2]), [0, 1, 1]
    )
    less_central = g.op("Less", own_depth, enemy_depth)
    equal_cyan = g.op(
        "And", g.op("Equal", own_depth, enemy_depth), g.op("Less", p[15], g.const(0.5))
    )
    close = g.op("Greater", distance_after, g.const(0.5))
    losing = g.op(
        "Cast",
        g.op("Or", close, g.op("Or", less_central, equal_cyan)),
        to=onnx.TensorProto.FLOAT,
    )
    # Actual Kernel capture is excluded from trigger and is never penalized.
    policies.append(g.scale(g.op("Mul", trigger, losing), -14.0 * strength))
    policies.append(
        g.scale(g.op("Mul", trigger, g.op("Sub", g.const(1.0), losing)), 4.0 * strength)
    )

    # Modest exchange compensation: permit a threatened piece to take a
    # sufficiently valuable target, especially with a friendly recapture.
    capture = g.op("Mul", enemy_nonking, g.reshape("target_map_1", [0, 1, 64]))
    defended = g.clip(attacks[0])
    exchange = g.op(
        "Mul", g.reshape(own_nonking, [0, 64, 1]), g.op("Mul", capture, defended)
    )
    policies.append(g.scale(exchange, 0.12 * strength))

    # Exact current-board single-friendly-blocker rays. Reward the first
    # action vacating an allied Ping/Daemon/Trace/Ice capture line so another
    # piece can use the second action. This is a prior, not a forced tactic:
    # the moving blocker may change occupancy elsewhere and MCTS checks it.
    edges = []
    orth = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    diag = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for typ, dirs, limit in [
        (0, diag, 2),
        (1, orth, 3),
        (3, diag, 3),
        (4, orth + diag, 2),
    ]:
        for source in range(64):
            row, col = divmod(source, 8)
            for dr, dc in dirs:
                for d in range(2, limit + 1):
                    tr, tc = row + dr * d, col + dc * d
                    if not (0 <= tr < 8 and 0 <= tc < 8):
                        continue
                    middle = [(row + dr * k) * 8 + col + dc * k for k in range(1, d)]
                    for blocker in middle:
                        others = [x for x in middle if x != blocker]
                        edges.append(
                            (
                                typ,
                                source,
                                tr * 8 + tc,
                                blocker,
                                others[0] if others else 0,
                                bool(others),
                            )
                        )
    edge = np.asarray(edges, np.int64)
    source_flat = g.reshape(
        g.op(
            "Gather",
            g.reshape("observation", [0, 1024]),
            g.const(edge[:, 0] * 64 + edge[:, 1], dtype=np.int64),
            axis=1,
        ),
        [0, len(edges)],
    )
    target = g.op(
        "Gather",
        g.reshape("target_map_1", [0, 64]),
        g.const(edge[:, 2], dtype=np.int64),
        axis=1,
    )
    blocker = g.op(
        "Gather",
        g.reshape(own_nonking, [0, 64]),
        g.const(edge[:, 3], dtype=np.int64),
        axis=1,
    )
    occupied = g.reshape(g.add(ownocc, enemyocc, p[12]), [0, 64])
    other = g.op("Gather", occupied, g.const(edge[:, 4], dtype=np.int64), axis=1)
    other = g.op("Mul", other, g.const(edge[:, 5].astype(np.float32)))
    open_other = g.clip(g.op("Sub", g.const(1.0), other))
    target_open = g.op(
        "Gather",
        g.reshape(g.op("Sub", g.const(1.0), p[12]), [0, 64]),
        g.const(edge[:, 2], dtype=np.int64),
        axis=1,
    )
    quality = g.op(
        "Mul",
        g.op("Mul", source_flat, target),
        g.op("Mul", blocker, g.op("Mul", open_other, target_open)),
    )
    scatter = np.zeros((len(edges), 64), np.float32)
    scatter[np.arange(len(edges)), edge[:, 3]] = 1.0
    available = g.reshape(g.op("MatMul", quality, g.const(scatter)), [0, 64, 1])
    two_left = g.op("Sub", g.scale(p[13], 2.0), g.const(1.0))
    cooperation = g.op("Mul", available, g.op("Mul", relocate, two_left))
    policies.append(g.scale(g.clip(cooperation, 0.0, 8.0), 0.25 * strength))
    return policies, values
