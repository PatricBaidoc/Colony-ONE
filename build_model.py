"""Optimize the supplied model11.onnx; no training framework or external weights.

The published game engine and MCTS remain unchanged. All added geometry is
inside the exported ONNX graph, so the submission is one portable model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as H
from onnx import numpy_helper as NH

ROOT = Path(__file__).resolve().parent
DEFAULTS = {
    "capture_scale": 0.7,
    "central_scale": 1.3,
    "material_scale": 1.3,
    "unsafe": 0.25,
    "guard_bonus": 0.06,
    "burn": 0.9,
    "attack": 0.35,
    "exact": True,
    "risk": 0.0,
    "policy_risk": 0.0,
    "kernel_policy_risk": 0.0,
    "tempo": 0.0,
    "pressure": 0.0,
    "force_win": False,
    "force_policy": 0.0,
    "piece_scales": None,
    "guard_policy": False,
    "followup": 0.0,
    "followup_last": 0.15,
    "side_strategy": 0.0,
    "cyan_policy_scale": 1.0,
    "border_safety": 0.0,
    "kernel_routes": 0.0,
    "protection_priority": 0.0,
    "evacuate_all": 0.0,
}


class Graph:
    def __init__(self, model):
        self.model = model
        self.nodes = []
        self.n = 0

    def const(self, array, name=None, dtype=np.float32):
        self.n += 1
        name = name or f"opt_c{self.n}"
        self.model.graph.initializer.append(
            NH.from_array(np.asarray(array, dtype=dtype), name)
        )
        return name

    def op(self, kind, *inputs, name=None, **attrs):
        self.n += 1
        name = name or f"opt_x{self.n}"
        self.nodes.append(H.make_node(kind, list(inputs), [name], **attrs))
        return name

    def scale(self, x, scale):
        return self.op("Mul", x, self.const(scale))

    def add(self, *xs):
        result = xs[0]
        for x in xs[1:]:
            result = self.op("Add", result, x)
        return result

    def reduce(self, x, axes, keepdims=0):
        return self.op(
            "ReduceSum", x, self.const(axes, dtype=np.int64), keepdims=keepdims
        )

    def unsqueeze(self, x, axes):
        return self.op("Unsqueeze", x, self.const(axes, dtype=np.int64))

    def reshape(self, x, shape, name=None):
        return self.op("Reshape", x, self.const(shape, dtype=np.int64), name=name)

    def clip(self, x, lo=0.0, hi=1.0):
        return self.op("Clip", x, self.const(lo), self.const(hi))


def geometry():
    """Capture geometry and intermediate squares, in canonical coordinates."""
    masks = np.zeros((6, 64, 64), np.float32)
    mid = np.zeros((2, 64, 64), np.int64)
    use = np.zeros((2, 64, 64), np.float32)
    shoot = np.zeros((6, 64, 64), np.float32)
    for f in range(64):
        for t in range(64):
            dr, dc = t // 8 - f // 8, t % 8 - f % 8
            a, b = abs(dr), abs(dc)
            d = max(a, b)
            orth = (dr == 0 or dc == 0) and d > 0
            diag = a == b and d > 0
            adj = d == 1
            masks[:, f, t] = [
                diag and d <= 2,
                adj or (orth and d <= 3),
                (a, b) in [(1, 2), (2, 1)],
                diag and 2 <= d <= 3,
                (orth or diag) and d <= 2,
                adj,
            ]
            shoot[1, f, t] = adj
            shoot[3, f, t] = diag and 2 <= d <= 3
            if (orth or diag) and d <= 3:
                for k in range(d - 1):
                    mid[k, f, t] = (
                        (f // 8 + np.sign(dr) * (k + 1)) * 8
                        + f % 8
                        + np.sign(dc) * (k + 1)
                    )
                    use[k, f, t] = 1
    return masks, mid, use, shoot


def kernel_route_terms(g, p, ownocc, enemyocc, attacks, relocate, adjacency, strength):
    """Prefer early King evacuation and let allies clear its inner escape doors."""
    adj = g.const(adjacency)
    ring = g.reshape("burning_ring", [0, 1, 64])
    inner = g.clip(g.op("Sub", g.const(1.0), g.add(p[12], ring)))
    urgency = g.clip(g.scale(g.op("Sub", p[14], g.const(0.25)), 2.0))
    policies, values = [], []
    room = []
    demands = []
    for king, friendly, hostile_attacks, redirect in [
        (p[5], ownocc, attacks[1], "enemy_guarded_marker"),
        (p[11], enemyocc, attacks[0], "own_guarded_marker"),
    ]:
        allowed = g.op("Mul", g.op("Sub", g.const(1.0), friendly), inner)
        allowed = g.op(
            "Mul", allowed, g.op("Sub", g.const(1.0), g.reshape(redirect, [0, 1, 64]))
        )
        safe = g.op("Mul", allowed, g.op("Sub", g.const(1.0), g.clip(hostile_attacks)))
        neighbors = g.op("MatMul", safe, adj)
        count = g.reduce(g.op("Mul", king, neighbors), [1, 2])
        room.append(g.reshape(count, [0, 1]))
        on_ring = g.reduce(g.op("Mul", king, ring), [1, 2])
        demand = g.op(
            "Mul", g.reshape(on_ring, [0, 1, 1]), g.reshape(urgency, [0, 1, 64])
        )
        demands.append(demand)
    current_demand = demands[0]
    king_neighbors = g.op("MatMul", p[5], adj)
    doors = g.op("Mul", king_neighbors, inner)
    blocker = g.op("Mul", g.op("Sub", ownocc, p[5]), doors)
    clear_door = g.op(
        "Mul", g.reshape(blocker, [0, 64, 1]), g.op("Sub", g.const(1.0), doors)
    )
    priority = g.add(g.const(0.35), g.scale(current_demand, 3.0))
    policies.append(
        g.scale(g.op("Mul", g.op("Mul", clear_door, relocate), priority), strength)
    )
    # Do not park Ice (or another ally) on a scarce inner escape door.
    scarce = g.op(
        "Cast",
        g.op("Less", g.reshape(room[0], [0, 1, 1]), g.const(2.0)),
        to=onnx.TensorProto.FLOAT,
    )
    block_door = g.op("Mul", g.reshape(g.op("Sub", ownocc, p[5]), [0, 64, 1]), doors)
    block_penalty = g.op(
        "Mul", g.op("Mul", block_door, relocate), g.op("Mul", priority, scarce)
    )
    policies.append(g.scale(block_penalty, -strength))
    # Reward King moves with onward room and early progress into the next
    # safe ring. Attacked destinations are penalized rather than forbidden.
    open_square = g.op(
        "Mul", g.op("Sub", g.const(1.0), ownocc), g.op("Sub", g.const(1.0), p[12])
    )
    open_square = g.op(
        "Mul",
        open_square,
        g.op("Sub", g.const(1.0), g.reshape("enemy_guarded_marker", [0, 1, 64])),
    )
    safe_square = g.op("Mul", open_square, g.op("Sub", g.const(1.0), g.clip(attacks[1])))
    onward = g.op("MatMul", safe_square, adj)
    king_source = g.reshape(p[5], [0, 64, 1])
    onward_delta = g.op("Sub", onward, g.reshape(onward, [0, 64, 1]))
    king_move = g.op("Mul", king_source, relocate)
    policies.append(g.scale(g.op("Mul", king_move, onward_delta), 0.16 * strength))
    evacuate = g.op("Mul", g.op("Mul", king_move, inner), current_demand)
    policies.append(g.scale(evacuate, 4.0 * strength))
    danger = g.op("Mul", king_move, g.clip(attacks[1]))
    policies.append(g.scale(danger, -2.0 * strength))
    depth = np.asarray(
        [min(i // 8, i % 8, 7 - i // 8, 7 - i % 8) for i in range(64)], np.float32
    )
    inward = g.const(depth[None, :] - depth[:, None])
    policies.append(g.scale(g.op("Mul", king_move, inward), 0.45 * strength))
    for sign, count, demand in [(-1.0, room[0], demands[0]), (1.0, room[1], demands[1])]:
        trapped = g.op(
            "Cast", g.op("Less", count, g.const(0.5)), to=onnx.TensorProto.FLOAT
        )
        urgent_trap = g.op("Mul", trapped, g.reshape(g.reduce(demand, [2]), [0, 1]))
        # Demand is broadcast over squares; normalize the reduction.
        values.append(g.scale(urgent_trap, sign * 1.5 * strength / 64.0))
    mobility = g.op("Sub", g.clip(room[0], 0.0, 3.0), g.clip(room[1], 0.0, 3.0))
    values.append(g.scale(mobility, 0.06 * strength))
    return policies, values


def protection_terms(
    g, p, ownocc, enemyocc, attacks, clear, relocate, masks, shots, cfg, force_target=None
):
    """Material safety and two-action threats, using Colony movement rules."""
    moves = np.zeros((6, 64, 64), np.float32)
    for f in range(64):
        for t in range(64):
            dr, dc = abs(t // 8 - f // 8), abs(t % 8 - f % 8)
            d = max(dr, dc)
            orth = (dr == 0 or dc == 0) and d > 0
            moves[:, f, t] = [
                orth and d <= 2,
                orth and d <= 3,
                (dr, dc) in [(1, 2), (2, 1)],
                orth and d == 1,
                0 < d <= 2 and (orth or dr == dc),
                d == 1,
            ]
    empty = g.clip(g.op("Sub", g.const(1.0), g.add(ownocc, enemyocc, p[12])))
    victims = g.op("Sub", ownocc, p[5])
    threats, guard_removals = [], []
    for typ in range(6):
        quiet = g.op("Mul", g.const(moves[typ]), empty)
        captures = g.op("Mul", g.const(masks[typ] * (1.0 - shots[typ])), victims)
        first = g.add(quiet, captures)
        if typ != 2:
            first = g.op("Mul", first, clear)
        first = g.op("Mul", first, g.reshape(p[6 + typ], [0, 64, 1]))
        arrivals = g.reduce(first, [1], keepdims=1)
        second = g.const(masks[typ])
        if typ != 2:
            second = g.op("Mul", second, clear)
        reach = g.op("Mul", g.reshape(arrivals, [0, 64, 1]), second)
        threats.append(g.reduce(reach, [1], keepdims=1))
        captured_guard = g.op("Mul", g.reshape(victims, [0, 64, 1]), g.const(masks[5]))
        guard_removals.append(
            g.reduce(g.op("Mul", reach, captured_guard), [1], keepdims=1)
        )
    two = g.clip(g.add(*threats))
    removed_guard = g.clip(g.add(*guard_removals))
    strength = cfg["protection_priority"]
    policies, values = [], []
    scales = np.asarray(cfg["piece_scales"] or [1.0] * 6, np.float32)
    costs = np.asarray([1.0, 1.8, 1.4, 1.3, 1.8, 0.0], np.float32) * scales
    material = g.add(*[g.scale(p[i], float(costs[i])) for i in range(6)])
    loss = g.op(
        "Relu",
        g.op(
            "Sub",
            g.reshape(material, [0, 64, 1]),
            g.scale(g.reshape("target_map_1", [0, 1, 64]), 0.5),
        ),
    )
    risk = g.add(g.clip(attacks[1]), g.scale(two, 0.35))
    defended = g.op("Sub", g.const(1.0), g.scale(g.clip(attacks[0]), 0.35))
    stage = g.op("Sub", g.const(1.5), p[13])
    exposure = g.op("Mul", g.op("Mul", loss, risk), g.op("Mul", relocate, defended))
    policies.append(g.scale(g.op("Mul", exposure, stage), -strength))
    guards = g.reshape("own_guards", [0, 1, 64])
    unguarded = g.op(
        "Cast", g.op("Less", guards, g.const(1.0)), to=onnx.TensorProto.FLOAT
    )
    one_guard = g.op(
        "Cast", g.op("Less", guards, g.const(2.0)), to=onnx.TensorProto.FLOAT
    )
    king_risk = g.add(
        g.op("Mul", g.clip(attacks[1]), one_guard),
        g.op("Mul", two, unguarded),
        g.op("Mul", removed_guard, one_guard),
    )
    king_exposure = g.op("Mul", g.reshape(p[5], [0, 64, 1]), g.clip(king_risk))
    policies.append(
        g.scale(g.op("Mul", g.op("Mul", king_exposure, relocate), stage), -3.0 * strength)
    )
    current_risk = g.reduce(g.op("Mul", p[5], g.clip(king_risk)), [1, 2])
    values.append(g.scale(g.reshape(current_risk, [0, 1]), -0.3 * strength))
    if cfg["evacuate_all"]:
        ring = g.reshape("burning_ring", [0, 1, 64])
        urgency = g.clip(g.scale(g.op("Sub", p[14], g.const(0.25)), 2.0))
        burning = g.op("Mul", ring, urgency)
        delta = g.op("Sub", g.reshape(burning, [0, 64, 1]), burning)
        # Even a piece that moves can still burn if its destination remains
        # on the next disappearing ring (Trace H7->H6 in match 18883).
        evacuate = g.op(
            "Mul", g.op("Mul", g.reshape(ownocc, [0, 64, 1]), delta), relocate
        )
        policies.append(g.scale(evacuate, 1.5 * cfg["evacuate_all"]))
        after = g.add(
            g.op("Mul", relocate, burning),
            g.op(
                "Mul", g.op("Sub", g.const(1.0), relocate), g.reshape(burning, [0, 64, 1])
            ),
        )
        staying = g.op(
            "Mul", g.reshape(g.add(material, g.scale(p[5], 5.0)), [0, 64, 1]), after
        )
        if force_target is not None:
            staying = g.op("Mul", staying, g.op("Sub", g.const(1.0), force_target))
        policies.append(g.scale(staying, -cfg["evacuate_all"]))
    return policies, values


def build(output, config=None, base=None):
    cfg = dict(DEFAULTS)
    if config:
        unknown = set(config) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown parameters: {unknown}")
        cfg.update(config)
    base = Path(base or ROOT / "baselines/model11.onnx")
    model = onnx.load(base)
    initializers = {t.name: t for t in model.graph.initializer}

    def set_weight(name, value):
        old = NH.to_array(initializers[name])
        value = np.broadcast_to(np.asarray(value, np.float32), old.shape).copy()
        initializers[name].CopyFrom(NH.from_array(value, name))

    target = NH.to_array(initializers["policy_target_weights"]).copy()
    target *= cfg["capture_scale"]
    piece_scales = np.asarray(cfg["piece_scales"] or [1.0] * 6, np.float32)
    if (
        piece_scales.shape != (6,)
        or not np.isfinite(piece_scales).all()
        or (piece_scales <= 0).any()
    ):
        raise ValueError("piece_scales must be six finite positive multipliers")
    target[0, 6:12, 0, 0] *= piece_scales
    set_weight("policy_target_weights", target)
    set_weight("enemy_kernel_guarded_discount", 5.3 * cfg["capture_scale"])
    set_weight(
        "movement_bias", NH.to_array(initializers["movement_bias"]) * cfg["central_scale"]
    )
    vw = NH.to_array(initializers["value_weights"]).copy().reshape(16, 64)
    vw[:12] *= cfg["material_scale"]
    vw[:6] *= piece_scales[:, None]
    vw[6:12] *= piece_scales[:, None]
    set_weight("value_weights", vw.reshape(1, -1))
    for name, value in [
        ("own_unsafe_scale", -cfg["unsafe"]),
        ("enemy_unsafe_scale", cfg["unsafe"]),
        ("own_guard2_scale", cfg["guard_bonus"]),
        ("enemy_guard2_scale", -cfg["guard_bonus"]),
        ("k_burn_scale", cfg["burn"]),
        ("k_attack_scale", cfg["attack"]),
    ]:
        set_weight(name, value)

    if cfg["exact"]:
        g = Graph(model)
        planes = g.reshape("observation", [0, 16, 64])
        p = [
            g.op("Gather", planes, g.const([i], dtype=np.int64), axis=1)
            for i in range(16)
        ]
        ownocc = g.add(*p[:6])
        enemyocc = g.add(*p[6:12])
        blockers = g.reshape(g.add(ownocc, enemyocc, p[12]), [0, 64])
        masks, mids, uses, shots = geometry()
        blocks = []
        for k in range(2):
            block = g.op("Gather", blockers, g.const(mids[k], dtype=np.int64), axis=1)
            blocks.append(g.op("Mul", block, g.const(uses[k])))
        clear = g.clip(g.op("Sub", g.const(1.0), g.add(*blocks)))
        attack_matrices = []
        for offset in [0, 6]:
            terms = []
            for typ in range(6):
                source = g.reshape(p[offset + typ], [0, 64, 1])
                term = g.op("Mul", source, g.const(masks[typ]))
                # Exploits ignore intervening pieces and locked squares.
                if typ != 2:
                    term = g.op("Mul", term, clear)
                terms.append(term)
            matrix = g.op("Mul", g.add(*terms), g.op("Sub", g.const(1.0), p[12]))
            attack_matrices.append(matrix)
        attacks = [g.reduce(x, [1], keepdims=1) for x in attack_matrices]
        for node in model.graph.node:
            if list(node.output) in [["own_attack"], ["enemy_attack"]]:
                node.output[0] += "_approx_unused"
        g.reshape(attacks[0], [0, 1, 8, 8], name="own_attack")
        g.reshape(attacks[1], [0, 1, 8, 8], name="enemy_attack")

        # An attack on a guarded Kernel leaves the attacker at its origin.
        # With two actions and at most one guard the same attack wins this
        # turn. Recognize it explicitly, including at uncached bodyguard
        # redirect leaves in the published MCTS implementation.
        force_target = None
        if cfg["force_win"] or cfg["force_policy"]:
            guards = g.reshape("enemy_guards", [0, 1, 64])
            actions = g.scale(g.reshape(p[13], [0, 1, 64]), 2.0)
            enough = g.op(
                "Cast", g.op("Less", guards, actions), to=onnx.TensorProto.FLOAT
            )
            force_target = g.op("Mul", enough, p[11])
            can_take = g.reduce(g.op("Mul", attacks[0], force_target), [1, 2])
            force_flag = g.op(
                "Greater",
                g.reshape(can_take, [0, 1]),
                g.const(0.0),
                name="opt_force_win_flag",
            )
            if cfg["force_win"]:
                for node in model.graph.node:
                    if list(node.output) == ["value_column"]:
                        node.output[0] = "value_column_static"
                g.op(
                    "Where",
                    force_flag,
                    g.const(0.999),
                    "value_column_static",
                    name="value_column",
                )

        extra_value = []
        if cfg["risk"]:
            # Static exchanges are tempo-sensitive: a piece attacked while
            # its owner is moving may retreat or take the attacking piece.
            material = np.asarray(
                [0.145, 0.298, 0.225, 0.201, 0.298, 0], np.float32
            ).reshape(1, 6, 1)
            ownmat = g.reduce(
                g.op(
                    "Mul",
                    g.op(
                        "Gather", planes, g.const(list(range(6)), dtype=np.int64), axis=1
                    ),
                    g.const(material),
                ),
                [1],
                keepdims=1,
            )
            enemymat = g.reduce(
                g.op(
                    "Mul",
                    g.op(
                        "Gather",
                        planes,
                        g.const(list(range(6, 12)), dtype=np.int64),
                        axis=1,
                    ),
                    g.const(material),
                ),
                [1],
                keepdims=1,
            )
            riskown = g.reduce(
                g.op("Mul", ownmat, g.clip(attacks[1])), [1, 2], keepdims=0
            )
            riskenemy = g.reduce(
                g.op("Mul", enemymat, g.clip(attacks[0])), [1, 2], keepdims=0
            )
            # Encourage threats on our turn; do not mistake every attack on
            # our movable piece for an inevitable capture.
            risknet = g.op("Sub", riskenemy, g.scale(riskown, 0.35))
            extra_value.append(g.scale(g.reshape(risknet, [0, 1]), cfg["risk"]))
        if cfg["tempo"]:
            ownhit = g.reduce(g.op("Mul", p[5], attacks[1]), [1, 2])
            enemyhit = g.reduce(g.op("Mul", p[11], attacks[0]), [1, 2])
            extra_value.append(
                g.scale(g.reshape(g.add(ownhit, enemyhit), [0, 1]), cfg["tempo"])
            )
        if cfg["pressure"]:
            # Safe control around the opposing Kernel, not global mobility.
            adj = np.zeros((64, 64), np.float32)
            for f in range(64):
                for t in range(64):
                    adj[f, t] = max(abs(f // 8 - t // 8), abs(f % 8 - t % 8)) <= 2
            near = [g.op("MatMul", p[i], g.const(adj)) for i in [5, 11]]
            press = g.op(
                "Sub",
                g.reduce(g.op("Mul", g.clip(attacks[0]), near[1]), [1, 2]),
                g.reduce(g.op("Mul", g.clip(attacks[1]), near[0]), [1, 2]),
            )
            extra_value.append(g.scale(g.reshape(press, [0, 1]), cfg["pressure"]))
        extra_policy = []
        if cfg["force_policy"]:
            extra_policy.append(g.scale(force_target, cfg["force_policy"]))
        if cfg["guard_policy"]:
            adjacent = np.zeros((64, 64), np.float32)
            for f in range(64):
                for t in range(64):
                    adjacent[f, t] = (
                        0 < max(abs(f // 8 - t // 8), abs(f % 8 - t % 8)) <= 1
                    )
            adj = g.const(adjacent)
            no_prior_guard = g.const(1.0)
            costs = []
            for typ in [0, 2, 3, 1, 4]:
                has = g.clip(g.op("MatMul", p[6 + typ], adj))
                selected = g.op("Mul", has, no_prior_guard)
                costs.append(g.scale(selected, float(target[0, 6 + typ, 0, 0])))
                no_prior_guard = g.op(
                    "Mul", no_prior_guard, g.op("Sub", g.const(1.0), has)
                )
            # The old base policy assigns every guarded Kernel the same 3.2.
            # Replace that with the cost of the guard actually sacrificed.
            actual_cost = g.add(*costs)
            old_guarded = float(target[0, 11, 0, 0]) - 5.3 * cfg["capture_scale"]
            correction = g.op(
                "Mul",
                g.op("Sub", actual_cost, g.const(old_guarded)),
                g.reshape("enemy_guarded_marker", [0, 1, 64]),
            )
            extra_policy.append(correction)
        if (
            cfg["policy_risk"]
            or cfg["kernel_policy_risk"]
            or cfg["followup"]
            or cfg["side_strategy"]
            or cfg["border_safety"]
            or cfg["kernel_routes"]
            or cfg["protection_priority"]
            or cfg["evacuate_all"]
        ):
            # Shooting and bodyguard redirection leave the attacker in place.
            shot_terms = []
            for typ in [1, 3]:
                shot_terms.append(
                    g.op("Mul", g.reshape(p[typ], [0, 64, 1]), g.const(shots[typ]))
                )
            shoot = g.op("Mul", g.add(*shot_terms), enemyocc)
            guarded = g.reshape("enemy_guarded_marker", [0, 1, 64])
            stationary = g.clip(g.add(shoot, guarded))
            relocate = g.op("Sub", g.const(1.0), stationary)
            if cfg["policy_risk"] or cfg["kernel_policy_risk"]:
                weights = g.add(
                    g.scale(ownocc, cfg["policy_risk"]),
                    g.scale(p[5], cfg["kernel_policy_risk"]),
                )
                weights = g.reshape(weights, [0, 64, 1])
                danger = g.op("Mul", weights, g.clip(attacks[1]))
                extra_policy.append(g.scale(g.op("Mul", danger, relocate), -1.0))
            if cfg["followup"]:
                # Rank quiet moves by the capture opportunities they create
                # for the SAME piece on the next action. Unlike a plain
                # source+target score this depends on the moving piece type.
                # The source is still occupied in this hypothetical board,
                # so newly opened rays are conservatively underestimated.
                values = g.reshape("target_map_1", [0, 64, 1])
                terms = []
                for typ in range(6):
                    reach = g.const(masks[typ])
                    if typ != 2:
                        reach = g.op("Mul", reach, clear)
                    opportunities = g.op("MatMul", reach, values)
                    delta = g.op(
                        "Sub", g.reshape(opportunities, [0, 1, 64]), opportunities
                    )
                    terms.append(g.op("Mul", g.reshape(p[typ], [0, 64, 1]), delta))
                continuation = g.add(*terms)
                has_second = g.op("Sub", g.scale(p[13], 2.0), g.const(1.0))
                tempo_weight = g.add(
                    g.const(cfg["followup_last"]),
                    g.scale(has_second, 1.0 - cfg["followup_last"]),
                )
                continuation = g.op("Mul", continuation, tempo_weight)
                continuation = g.op("Mul", continuation, relocate)
                extra_policy.append(g.scale(continuation, cfg["followup"]))
            if cfg["side_strategy"]:
                # Magenta can win rule tiebreaks, but still takes any forced
                # Kernel capture. Be conservative about exposed destinations
                # and dismantling its own Kernel's cover.
                passive = g.scale(p[15], cfg["side_strategy"])
                safety = g.op("Mul", g.reshape(ownocc, [0, 64, 1]), g.clip(attacks[1]))
                kernel_safety = g.op(
                    "Mul", g.reshape(p[5], [0, 64, 1]), g.clip(attacks[1])
                )
                caution = g.op(
                    "Mul",
                    g.add(safety, g.scale(kernel_safety, 1.5)),
                    g.op("Mul", relocate, passive),
                )
                extra_policy.append(g.scale(caution, -0.8))
                near_kernel = g.reshape("own_k_3x3", [0, 1, 64])
                guard_delta = g.op("Sub", near_kernel, g.reshape("own_k_3x3", [0, 64, 1]))
                guards = g.reshape(g.op("Sub", ownocc, p[5]), [0, 64, 1])
                retain_cover = g.op(
                    "Mul",
                    g.op("Mul", guards, guard_delta),
                    g.op("Mul", relocate, passive),
                )
                extra_policy.append(g.scale(retain_cover, 0.6))
            if cfg["border_safety"]:
                # A shot or redirected guard capture does not evacuate its
                # source square. Only relocation receives an escape bonus.
                burning = g.reshape("urgent_burn", [0, 1, 64])
                burn_delta = g.op("Sub", g.reshape("urgent_burn", [0, 64, 1]), burning)
                evacuate = g.op("Mul", burn_delta, relocate)
                extra_policy.append(g.scale(evacuate, 0.8 * cfg["border_safety"]))
                kernel_burn_dest = g.op("Mul", g.reshape(p[5], [0, 64, 1]), burning)
                extra_policy.append(
                    g.scale(
                        g.op("Mul", kernel_burn_dest, relocate),
                        -3.0 * cfg["border_safety"],
                    )
                )
                material = g.op("Sub", ownocc, enemyocc)
                exposed = g.reduce(g.op("Mul", material, burning), [1, 2])
                extra_value.append(
                    g.scale(g.reshape(exposed, [0, 1]), -0.2 * cfg["border_safety"])
                )
            if cfg["kernel_routes"]:
                route_policy, route_value = kernel_route_terms(
                    g,
                    p,
                    ownocc,
                    enemyocc,
                    attacks,
                    relocate,
                    masks[5],
                    cfg["kernel_routes"],
                )
                extra_policy.extend(route_policy)
                extra_value.extend(route_value)
            if cfg["protection_priority"] or cfg["evacuate_all"]:
                protect_policy, protect_value = protection_terms(
                    g,
                    p,
                    ownocc,
                    enemyocc,
                    attacks,
                    clear,
                    relocate,
                    masks,
                    shots,
                    cfg,
                    force_target,
                )
                extra_policy.extend(protect_policy)
                extra_value.extend(protect_value)

        if extra_value:
            for node in model.graph.node:
                if list(node.output) == ["value_logits"]:
                    node.output[0] = "value_logits_base"
            g.op(
                "Identity", g.add("value_logits_base", *extra_value), name="value_logits"
            )
        if extra_policy or cfg["cyan_policy_scale"] != 1.0:
            for node in model.graph.node:
                if list(node.output) == ["policy_grid"]:
                    node.output[0] = "policy_grid_base"
            policy_base = "policy_grid_base"
            if cfg["cyan_policy_scale"] != 1.0:
                factor = g.add(
                    g.const(cfg["cyan_policy_scale"]),
                    g.scale(p[15], 1.0 - cfg["cyan_policy_scale"]),
                )
                policy_base = g.op("Mul", policy_base, factor)
            g.op("Identity", g.add(policy_base, *extra_policy), name="policy_grid")
        # Topological order: extension depends on original guard planes, while
        # the original value head now depends on the extension attack planes.
        pending = list(model.graph.node) + g.nodes
        known = {i.name for i in model.graph.initializer} | {"observation"}
        ordered = []
        while pending:
            ready = [n for n in pending if all(i in known or not i for i in n.input)]
            if not ready:
                raise ValueError("Cycle or missing graph input")
            for n in ready:
                ordered.append(n)
                known.update(n.output)
                pending.remove(n)
        del model.graph.node[:]
        model.graph.node.extend(ordered)
        # Remove unused nodes/weights, including the old approximate attacks.
        needed = {"policy", "value"}
        kept = []
        for n in reversed(ordered):
            if any(x in needed for x in n.output):
                kept.append(n)
                needed.update(n.input)
        del model.graph.node[:]
        model.graph.node.extend(reversed(kept))
        kept_weights = [i for i in model.graph.initializer if i.name in needed]
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept_weights)

    model.producer_name = "colony-one-model11-optimized"
    model.graph.name = "model11_optimized_level4"
    meta = model.metadata_props.add()
    meta.key = "optimization_config"
    meta.value = json.dumps(cfg, sort_keys=True)
    onnx.checker.check_model(model)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output)
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "model.onnx")
    p.add_argument("--base", type=Path, default=ROOT / "baselines/model11.onnx")
    p.add_argument("--config", type=Path)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text()) if args.config else None
    print(json.dumps(build(args.output, cfg, args.base), indent=2))
    print(f"Created {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
