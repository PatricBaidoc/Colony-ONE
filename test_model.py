"""Check ONNX features against legal captures on actual engine positions."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto
from onnx import helper as H

from build_model import build
from colony_one import constants, encoding, mcts
from colony_one.board import Board
from colony_one.env import ColonyOneEnv
from colony_one.onnx_runner import OnnxEvaluator
from colony_one.rules import legal_actions_for_team
from metrics import glicko2

ROOT = Path(__file__).resolve().parent


class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "candidates/test_features.onnx"
        build(
            cls.path,
            {
                "risk": 0.5,
                "policy_risk": 1.0,
                "kernel_policy_risk": 1.0,
                "tempo": 0.3,
                "pressure": 0.03,
            },
        )
        model = onnx.load(cls.path)
        for name in ["own_attack", "enemy_attack"]:
            model.graph.output.append(
                H.make_tensor_value_info(name, TensorProto.FLOAT, ["batch", 1, 8, 8])
            )
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        cls.session = ort.InferenceSession(
            model.SerializeToString(), so, providers=["CPUExecutionProvider"]
        )

    def compare(self, env):
        obs = env.observation()
        _, _, own, enemy = self.session.run(None, {"observation": obs[None]})
        teams = [
            env.current_team,
            constants.MAGENTA if env.current_team == constants.CYAN else constants.CYAN,
        ]
        for team, actual in zip(teams, [own, enemy], strict=True):
            expected = np.zeros((8, 8), np.float32)
            for action in legal_actions_for_team(env.board, team):
                target = env.board.piece_at(*action.to_pos)
                if target is not None and target.team != team:
                    r, c = encoding.canonical_pos(action.to_pos, env.current_team)
                    expected[r, c] += 1
            for row in range(8):
                for col in range(8):
                    target = env.board.piece_at(row, col)
                    if target is not None and target.team != team:
                        r, c = encoding.canonical_pos((row, col), env.current_team)
                        self.assertEqual(
                            expected[r, c],
                            actual[0, 0, r, c],
                            (team, (row, col), env.render()),
                        )

    def test_capture_geometry_matches_engine(self):
        import json

        scene = np.asarray(
            json.loads((ROOT / "scenes/level4.json").read_text()), np.float32
        )
        checked = 0
        for seed in range(8):
            env = ColonyOneEnv(seed=seed, scene=scene)
            while not env.terminated:
                self.compare(env)
                checked += 1
                env.step(env.sample_legal_action())
        self.assertGreater(checked, 150)

    def test_dynamic_batches_and_extremes(self):
        for batch in [1, 2, 8, 16]:
            for scalar in [0.0, 1.0, -1.0, 2.0]:
                obs = np.full((batch, 16, 8, 8), scalar, np.float32)
                policy, value, *_ = self.session.run(None, {"observation": obs})
                self.assertEqual(policy.shape, (batch, 4096))
                self.assertEqual(value.shape, (batch,))
                self.assertTrue(np.isfinite(policy).all())
                self.assertTrue(np.isfinite(value).all())
                self.assertTrue((abs(value) <= 1).all())

    def test_glicko_published_example(self):
        result = glicko2(
            1500.0,
            200.0,
            0.06,
            [(1400.0, 30.0, 1.0), (1550.0, 100.0, 0.0), (1700.0, 300.0, 0.0)],
        )
        # The published worked example rounds intermediate g/E values.
        self.assertAlmostEqual(result["rating"], 1464.06, delta=0.02)
        self.assertAlmostEqual(result["rd"], 151.52, places=2)
        self.assertAlmostEqual(result["volatility"], 0.059996, places=5)

    def test_kernel_ice_blocker_cleared_before_firewall(self):
        """Only clearing Ice first permits a King escape in the current turn."""
        evaluator = OnnxEvaluator(ROOT / "model11_optimized.onnx")
        for locked_rings, round_number in [(0, 8), (1, 16)]:
            with self.subTest(locked_rings=locked_rings):
                scene = np.zeros((16, 8, 8), np.float32)
                low, high = locked_rings, 7 - locked_rings
                scene[5, high, low] = 1
                scene[4, high - 1, low + 1] = 1  # Ice blocks the only inner door.
                scene[0, high, low + 1] = 1
                scene[3, high - 1, low] = 1
                # Opponent already occupies the safe interior; waiting for
                # both Kernels to burn cannot give Magenta a tiebreak win.
                scene[11, 2, 5] = 1
                scene[6, 2, 4] = 1
                for row in range(8):
                    for col in range(8):
                        scene[12, row, col] = (
                            min(row, col, 7 - row, 7 - col) < locked_rings
                        )
                scene[13] = 1
                scene[14] = 1
                scene[15] = 1
                env = ColonyOneEnv(scene=scene)
                env.board = Board.from_observation(scene, team=constants.MAGENTA)
                env.current_team = constants.MAGENTA
                env.round_number = round_number
                config = mcts.MCTSConfig(simulations=100)
                rng = np.random.default_rng(121 + locked_rings)
                self.assertFalse(
                    any(
                        env.board.piece_at(*a.from_pos).piece_type.value == "KERNEL"
                        for a in env.legal_actions()
                    )
                )
                first = mcts.search_action(
                    env.board,
                    env.current_team,
                    env.actions_left,
                    env.round_number,
                    evaluator,
                    config,
                    rng=rng,
                )
                self.assertEqual(
                    env.board.piece_at(*first.from_pos).piece_type.value, "ICE"
                )
                env.step(first)
                second = mcts.search_action(
                    env.board,
                    env.current_team,
                    env.actions_left,
                    env.round_number,
                    evaluator,
                    config,
                    rng=rng,
                )
                self.assertEqual(
                    env.board.piece_at(*second.from_pos).piece_type.value, "KERNEL"
                )
                env.step(second)
                self.assertFalse(env.terminated)
                own_kernel = [
                    (r, c)
                    for r in range(8)
                    for c in range(8)
                    if (p := env.board.piece_at(r, c)) is not None
                    and p.team == constants.MAGENTA
                    and p.piece_type.value == "KERNEL"
                ]
                self.assertEqual(len(own_kernel), 1)
                row, col = own_kernel[0]
                self.assertGreaterEqual(min(row, col, 7 - row, 7 - col), locked_rings + 1)
                self.assertTrue(env.legal_actions())

    def test_two_action_guard_win(self):
        path = ROOT / "candidates/test_force.onnx"
        build(path, {"force_win": True, "force_policy": 4.0})
        model = onnx.load(path)
        model.graph.output.append(
            H.make_tensor_value_info("opt_force_win_flag", TensorProto.BOOL, ["batch", 1])
        )
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        session = ort.InferenceSession(
            model.SerializeToString(), so, providers=["CPUExecutionProvider"]
        )
        # Daemon is adjacent to the opposing Kernel; Ping is its only guard.
        scene = np.zeros((16, 8, 8), np.float32)
        scene[5, 7, 0] = 1
        scene[1, 2, 3] = 1
        scene[11, 2, 2] = 1
        scene[6, 1, 1] = 1
        scene[13] = 1
        scene[14] = 0.125
        env = ColonyOneEnv(scene=scene)
        _, value, flag = session.run(None, {"observation": env.observation()[None]})
        self.assertTrue(flag[0, 0])
        self.assertAlmostEqual(float(value[0]), 0.999, places=5)
        attack = (2 * 8 + 3) * 64 + (2 * 8 + 2)
        env.step(attack)
        self.assertFalse(env.terminated)
        self.assertEqual(env.actions_left, 1)
        _, value, flag = session.run(None, {"observation": env.observation()[None]})
        self.assertTrue(flag[0, 0])
        env.step(attack)
        self.assertTrue(env.terminated)
        self.assertEqual(env.winner, constants.CYAN)
        # With one action remaining and a live guard, no forced capture this turn.
        env = ColonyOneEnv(scene=scene)
        env.actions_left = 1
        _, _, flag = session.run(None, {"observation": env.observation()[None]})
        self.assertFalse(flag[0, 0])

    def test_followup_prior_respects_remaining_actions(self):
        scene = np.zeros((16, 8, 8), np.float32)
        scene[5, 7, 0] = 1
        scene[1, 3, 0] = 1
        scene[11, 3, 4] = 1
        scene[13] = 1
        scene[14] = 0.125
        env = ColonyOneEnv(scene=scene)
        sessions = []
        for bonus in [0.0, 0.35]:
            path = ROOT / f"candidates/test_follow_{bonus}.onnx"
            build(path, {"capture_scale": 1.0, "followup": bonus})
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1
            so.inter_op_num_threads = 1
            sessions.append(
                ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
            )
        action = (3 * 8) * 64 + (3 * 8 + 1)
        increments = []
        for actions_left in [2, 1]:
            env.actions_left = actions_left
            obs = env.observation()[None]
            outputs = [s.run(None, {"observation": obs})[0][0, action] for s in sessions]
            increments.append(float(outputs[1] - outputs[0]))
        self.assertAlmostEqual(increments[0], 0.35 * 8.5, places=5)
        self.assertAlmostEqual(increments[1], increments[0] * 0.15, places=5)
        # The suggested preparation really permits a winning second action.
        env.actions_left = 2
        env.step(action)
        env.step((3 * 8 + 1) * 64 + (3 * 8 + 4))
        self.assertTrue(env.terminated)
        self.assertEqual(env.winner, constants.CYAN)

    def test_firewall_evacuation_and_magenta_caution(self):
        path = ROOT / "candidates/test_strategy.onnx"
        build(
            path,
            {
                "side_strategy": 1.0,
                "cyan_policy_scale": 0.8,
                "border_safety": 1.0,
                "followup": 0.7,
            },
        )
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        session = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        scene = np.zeros((16, 8, 8), np.float32)
        scene[5, 7, 0] = 1
        scene[0, 7, 1] = 1
        scene[11, 0, 7] = 1
        scene[13] = 1
        scene[14] = 1
        env = ColonyOneEnv(scene=scene)
        env.round_number = 8
        policy = session.run(None, {"observation": env.observation()[None]})[0][0]
        inside = (7 * 8) * 64 + (6 * 8 + 1)
        edge = (7 * 8) * 64 + (6 * 8)
        self.assertIn(inside, env.legal_action_indices())
        self.assertIn(edge, env.legal_action_indices())
        self.assertGreater(policy[inside] - policy[edge], 2.5)
        env.step(inside)
        env.board.lock_outer_ring()
        self.assertIsNotNone(env.board.piece_at(6, 1))
        # On the same canonical position Magenta penalizes an exposed quiet
        # move more than Cyan; neither move captures an enemy.
        scene = np.zeros((16, 8, 8), np.float32)
        scene[5, 7, 0] = 1
        scene[4, 6, 1] = 1
        scene[7, 3, 1] = 1
        scene[11, 0, 7] = 1
        scene[13] = 1
        scene[14] = 0.125
        differences = []
        for side in [0.0, 1.0]:
            scene[15] = side
            policy = session.run(None, {"observation": scene[None]})[0][0]
            danger = (6 * 8 + 1) * 64 + (4 * 8 + 1)
            safe = (6 * 8 + 1) * 64 + (6 * 8 + 2)
            differences.append(float(policy[danger] - policy[safe]))
        self.assertLess(differences[1], differences[0] - 0.3)


if __name__ == "__main__":
    unittest.main()
