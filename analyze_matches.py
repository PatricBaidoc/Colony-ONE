"""Replay provided match protocols; determine results from winner and user side."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from colony_one import encoding
from colony_one.env import ColonyOneEnv

ROOT = Path(__file__).resolve().parent
MOVE = re.compile(
    r"^\s*(\d+)\.\s+(CYAN|MAGENTA)\s+\w+\s+([A-H][1-8])\s+->\s+([A-H][1-8])", re.M
)


def pos(s):
    return 8 - int(s[1]), ord(s[0]) - ord("A")


def analyze(path):
    records = json.loads(Path(path).read_text())
    scene = np.asarray(json.loads((ROOT / "scenes/level4.json").read_text()), np.float32)
    games = []
    for record in records:
        protocol = record.get("protocol", "")
        winners = re.findall(r"\*\*\* (CYAN|MAGENTA) WINS", protocol)
        if not winners:
            games.append({"match_id": record["match_id"], "unavailable": True})
            continue
        winner = winners[-1]
        entry = {
            "match_id": record["match_id"],
            "side": record["you"],
            "winner": winner,
            "won": winner == record["you"],
            "score_field": record["score"],
            "opponent": record.get("opponent"),
        }
        env = ColonyOneEnv(scene=scene)
        moves = list(MOVE.finditer(protocol))
        try:
            for move in moves:
                number, team, f, t = move.groups()
                if env.action != int(number) or env.current_team != team:
                    raise ValueError("Protocol turn disagrees with engine")
                fr, fc = encoding.canonical_pos(pos(f), team)
                tr, tc = encoding.canonical_pos(pos(t), team)
                action = (fr * 8 + fc) * 64 + tr * 8 + tc
                env.step(action)
            if not env.terminated or env.winner != winner:
                raise ValueError("Protocol winner disagrees with replay")
            kernel_counts = sum(
                p is not None and p.piece_type.value == "KERNEL"
                for row in env.board.grid
                for p in row
            )
            entry.update(
                actions=env.action - 1,
                replay_verified=True,
                terminal_kind="bare_kernels"
                if kernel_counts == 2
                else "firewall"
                if "FIREWALL" in protocol[moves[-1].end() :]
                else "kernel_capture",
            )
        except (ValueError, RuntimeError) as exc:
            entry.update(replay_verified=False, error=str(exc))
        games.append(entry)
    available = [g for g in games if not g.get("unavailable")]
    wins = sum(g["won"] for g in available)
    result = {
        "total": len(records),
        "protocols_available": len(available),
        "unavailable": len(records) - len(available),
        "wins_from_protocol": wins,
        "losses_from_protocol": len(available) - wins,
        "winrate_from_protocol": wins / len(available),
        "replay_verified": sum(g.get("replay_verified", False) for g in games),
        "score_field_interpretation": (
            "Every available score matches 1 for CYAN win, 0 for MAGENTA win, "
            "not the user outcome."
        ),
        "score_matches_cyan_winner": all(
            g["score_field"] == int(g["winner"] == "CYAN") for g in available
        ),
        "by_side": {
            side: {
                "games": sum(g["side"] == side for g in available),
                "wins": sum(g["side"] == side and g["won"] for g in available),
            }
            for side in ["CYAN", "MAGENTA"]
        },
        "loss_causes": dict(
            Counter(
                g.get("terminal_kind", "unverified") for g in available if not g["won"]
            )
        ),
        "games": games,
    }
    return result


if __name__ == "__main__":
    result = analyze(ROOT / "results/provided_matches.json")
    (ROOT / "results/provided_analysis.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "games"}, indent=2))
