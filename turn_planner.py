"""Short King routes and the exact remaining own-action firewall budget."""

import numpy as np
import onnx


def turn_planner(
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
    adj = g.const(masks[5])
    ring = g.reshape("burning_ring", [0, 1, 64])
    goal = g.clip(g.op("Sub", g.const(1.0), g.add(p[12], ring)))
    nonking = g.op("Sub", ownocc, p[5])
    walk = g.clip(
        g.op(
            "Sub",
            g.const(1.0),
            g.add(p[12], nonking, g.reshape("enemy_guarded_marker", [0, 1, 64])),
        )
    )
    reached = [g.op("Mul", goal, walk)]
    for _ in range(3):
        nxt = g.op("Mul", walk, g.clip(g.op("MatMul", reached[-1], adj)))
        reached.append(g.op("Max", reached[-1], nxt))
    distance = g.add(*[g.op("Sub", g.const(1.0), x) for x in reached])
    # An adjacent friendly piece can be moved first, then King follows. Only
    # count it if the piece has a legal relocating action away from the door.
    quiet = np.zeros((6, 64, 64), np.float32)
    for f in range(64):
        for t in range(64):
            dr, dc = abs(t // 8 - f // 8), abs(t % 8 - f % 8)
            d = max(dr, dc)
            orth = (dr == 0 or dc == 0) and d > 0
            quiet[:, f, t] = [
                orth and d <= 2,
                orth and d <= 3,
                (dr, dc) in [(1, 2), (2, 1)],
                orth and d == 1,
                0 < d <= 2 and (orth or dr == dc),
                d == 1,
            ]
    empty = g.clip(g.op("Sub", g.const(1.0), g.add(ownocc, enemyocc, p[12])))
    captures = g.op("Sub", enemyocc, g.reshape("enemy_guarded_marker", [0, 1, 64]))
    doors = g.op("Mul", g.op("MatMul", p[5], adj), goal)
    move_counts = []
    for typ in range(5):
        moves = g.add(
            g.op("Mul", g.const(quiet[typ]), empty),
            g.op("Mul", g.const(masks[typ] * (1.0 - shots[typ])), captures),
        )
        if typ != 2:
            moves = g.op("Mul", moves, clear)
        moves = g.op("Mul", moves, g.op("Sub", g.const(1.0), doors))
        count = g.reduce(moves, [2], keepdims=0)
        move_counts.append(g.op("Mul", p[typ], g.reshape(g.clip(count), [0, 1, 64])))
    movable = g.clip(g.add(*move_counts))
    unblock_goal = g.op("Mul", g.op("Mul", goal, nonking), movable)
    unblock_path = g.clip(g.op("MatMul", unblock_goal, adj))
    # Min(distance, 2) where an ally can clear the neighboring safe square.
    distance = g.op(
        "Min", distance, g.op("Sub", g.const(4.0), g.scale(unblock_path, 2.0))
    )
    current = g.reshape(g.reduce(g.op("Mul", p[5], distance), [1, 2]), [0, 1, 1])
    rounds = g.scale(g.op("Sub", g.const(1.0), p[14]), 8.0)
    budget = g.add(g.scale(rounds, 2.0), g.scale(p[13], 2.0))
    after = g.op("Max", g.const(0.0), g.op("Sub", budget, g.const(1.0)))
    king_ring = g.reshape(g.reduce(g.op("Mul", p[5], ring), [1, 2]), [0, 1, 1])
    imminent = g.op(
        "Cast", g.op("LessOrEqual", budget, g.const(3.0)), to=onnx.TensorProto.FLOAT
    )
    critical = g.op(
        "Cast", g.op("GreaterOrEqual", current, budget), to=onnx.TensorProto.FLOAT
    )
    critical = g.op("Mul", g.op("Mul", critical, imminent), king_ring)
    king_source = g.reshape(p[5], [0, 64, 1])
    advance = g.op("Cast", g.op("Less", distance, current), to=onnx.TensorProto.FLOAT)
    advance = g.op("Mul", g.op("Mul", king_source, advance), relocate)
    blockers = g.reshape(g.op("Mul", nonking, doors), [0, 64, 1])
    clear_door = g.op(
        "Mul", g.op("Mul", blockers, relocate), g.op("Sub", g.const(1.0), doors)
    )
    helpful = g.clip(g.add(advance, clear_door))
    exemption = (
        g.const(1.0) if force_target is None else g.op("Sub", g.const(1.0), force_target)
    )
    optional = g.op(
        "Mul", g.op("Mul", critical, g.op("Sub", g.const(1.0), helpful)), exemption
    )
    too_late = g.op("Cast", g.op("Greater", distance, after), to=onnx.TensorProto.FLOAT)
    losing_step = g.op(
        "Mul",
        g.op("Mul", king_source, relocate),
        g.op("Mul", g.op("Mul", too_late, imminent), exemption),
    )
    policies = [g.scale(optional, -7.0 * strength), g.scale(losing_step, -7.0 * strength)]
    progress = g.op("Sub", current, distance)
    policies.append(
        g.scale(
            g.op("Mul", g.op("Mul", king_source, progress), relocate), 0.35 * strength
        )
    )
    impossible = g.op("Cast", g.op("Greater", current, budget), to=onnx.TensorProto.FLOAT)
    bad = g.op("Mul", g.op("Mul", impossible, imminent), king_ring)
    value = g.reshape(g.reduce(bad, [1, 2]), [0, 1])
    return policies, [g.scale(value, -0.8 * strength / 64.0)]
