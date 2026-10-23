"""Glicko-2 with explicit rating-period assumptions; never a server rating claim."""

from __future__ import annotations

import math

SCALE = 173.7178


def glicko2(rating=1500.0, rd=350.0, volatility=0.06, opponents=(), tau=0.5):
    """One Glicko-2 period. Opponents: (rating, RD, score) tuples.

    Based on Mark Glickman's published Glicko-2 algorithm. Unknown server
    RD/volatility/period rules cannot be inferred from a winrate alone.
    """
    mu, phi = (rating - 1500) / SCALE, rd / SCALE
    if not opponents:
        return {
            "rating": rating,
            "rd": SCALE * math.sqrt(phi * phi + volatility * volatility),
            "volatility": volatility,
        }
    variance_inv = 0.0
    gradient = 0.0
    for opponent_rating, opponent_rd, score in opponents:
        g = 1 / math.sqrt(1 + 3 * (opponent_rd / SCALE) ** 2 / math.pi**2)
        expected = 1 / (1 + math.exp(-g * (mu - (opponent_rating - 1500) / SCALE)))
        variance_inv += g * g * expected * (1 - expected)
        gradient += g * (score - expected)
    variance = 1 / variance_inv
    delta = variance * gradient
    a = math.log(volatility * volatility)

    def f(x):
        ex = math.exp(x)
        return ex * (delta * delta - phi * phi - variance - ex) / (
            2 * (phi * phi + variance + ex) ** 2
        ) - (x - a) / (tau * tau)

    left = a
    if delta * delta > phi * phi + variance:
        right = math.log(delta * delta - phi * phi - variance)
    else:
        k = 1
        while f(a - k * tau) < 0:
            k += 1
        right = a - k * tau
    fl, fr = f(left), f(right)
    for _ in range(1000):
        if abs(right - left) <= 1e-7:
            break
        middle = left + (left - right) * fl / (fr - fl)
        fm = f(middle)
        if fm * fr <= 0:
            left, fl = right, fr
        else:
            fl /= 2
        right, fr = middle, fm
    else:
        raise RuntimeError("Glicko-2 volatility solver did not converge")
    new_sigma = math.exp(left / 2)
    phi_star = math.sqrt(phi * phi + new_sigma * new_sigma)
    new_phi = 1 / math.sqrt(1 / (phi_star * phi_star) + 1 / variance)
    new_mu = mu + new_phi * new_phi * gradient
    return {
        "rating": 1500 + SCALE * new_mu,
        "rd": SCALE * new_phi,
        "volatility": new_sigma,
    }


def local_glicko(records):
    result = glicko2(opponents=[(1500.0, 50.0, float(x["a_won"])) for x in records])
    result["assumptions"] = (
        "One period; candidate 1500/RD350/sigma0.06, fixed reference 1500/RD50, "
        "tau0.5. Not live tournament Glicko-2."
    )
    return result
