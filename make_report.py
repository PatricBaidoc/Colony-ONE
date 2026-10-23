"""Generate the deliverable report from completed, saved match results."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    reports = []
    for name in ["model11", "v2", "claude2"]:
        data = json.loads((ROOT / f"results/final_{name}.json").read_text())
        reports.append((name, data))
    primary = reports[0][1]["summary"]
    cfg = json.loads((ROOT / "best_config.json").read_text())
    model = ROOT / "model11_optimized.onnx"
    sha = hashlib.sha256(model.read_bytes()).hexdigest()
    plan = json.loads((ROOT / "results/final_test_plan.json").read_text())
    assert sha == plan["model_sha256"]
    assert all(data["sha256_a"] == sha for _, data in reports)
    assert "ACCEPTED --" in (ROOT / "results/validation.txt").read_text()
    rows = []
    for name, data in reports:
        s = data["summary"]
        lo, hi = s["wilson95"]
        rows.append(
            f"| {name} | {s['wins']}/{s['games']} | {s['winrate']:.1%} "
            f"| {lo:.1%}–{hi:.1%} |"
        )
    table = "\n".join(rows)
    by = primary["by_side"]
    g = primary["local_glicko2"]
    body = f"""# Model11 optimizat: strategie și rezultate

## Fișierele de folosit

- `model11_optimized.onnx`: **{model.stat().st_size:,} bytes**, un singur fișier
  cu toate ponderile incluse. Validatorul oficial: **ACCEPTED**.
- `main.py`: reconstruiește modelul ales din `best_config.json` și originalul
  `baselines/model11.onnx`. `build_model.py` conține optimizarea efectivă.
- `arena.py`, `metrics.py`: comparații reproductibile și Glicko-2 local.
- `results/`: rezultate brute, configurații de selecție și verificări.

SHA-256 model final: `{sha}`.

## Strategia cerută

**Magenta prudentă:** evită destinațiile atacate și mutările care îndepărtează
gărzile de propriul Kernel. Se păstrează presiunea cu Trace și Daemon fără
expunere inutilă. Stilul prudent nu împiedică o captură forțată a Kernelului.

**Cyan activă și variată:** priorii de bază sunt mai puțin concentrați, iar
mutările care pregătesc o captură cu aceeași piesă la a doua acțiune primesc
bonus. Variația aleatoare din deschidere este cea oficială a arbitrului;
modelul ONNX este determinist.

**Kernelul adversar rămâne obiectivul:** evaluatorul recunoaște atacul repetat
care sacrifică întâi unica gardă și capturează Kernelul la a doua acțiune.
Atacurile țin cont de blocaje, salturi și distanțele speciale de tragere.

**Firewall:** evaluăm următorul inel care urmează să dispară, după fiecare
restrângere. Cu cât închiderea este mai aproape, cu atât cresc bonusul de
evacuare și penalizarea destinațiilor condamnate. Penalizarea este mult mai
mare pentru Kernel. Tragerea și capturarea unei gărzi fără relocare nu sunt
confundate cu evacuarea atacatorului.

Aceasta este optimizare euristică și parametrică a model11, nu un model
AlphaZero antrenat pe GPU. Motorul și MCTS-ul original au rămas neschimbate.

## Verificarea finală, pe semințe nefolosite la selecție

Scena **Level 4**, **100 simulări MCTS per acțiune**, noise și sampling oficial,
culori alternate în perechi. Configurația a fost înghețată înainte de aceste
teste; planul, semințele și hash-urile sunt în `results/final_test_plan.json`.

| Adversar din arhiva primită | Victorii | Winrate | Interval Wilson 95% |
|---|---:|---:|---:|
{table}

Împotriva **model11 original**, pe culori:

- Cyan: **{by["CYAN"]["wins"]}/{by["CYAN"]["games"]}**.
- Magenta: **{by["MAGENTA"]["wins"]}/{by["MAGENTA"]["games"]}**.

Bootstrap pe perechi, 95%: **{primary["paired_bootstrap95"][0]:.1%}–{primary["paired_bootstrap95"][1]:.1%}**.
Diferență Elo descriptivă cu regularizare: **{primary["elo_difference_smoothed"]:+.1f}**.
Glicko-2 local: **{g["rating"]:.1f}**, RD **{g["rd"]:.1f}**, volatilitate
**{g["volatility"]:.6f}**. Ipoteze: o singură perioadă, modelul nou începe la
1500/RD350, originalul este referință fixă 1500/RD50, sigma inițial 0.06,
tau 0.5. **Acesta nu este ratingul live al turneului.** Lipsesc ratingurile,
RD-urile și perioadele de rating folosite de server pentru adversarii reali.

Rezultatele sunt locale, împotriva modelelor disponibile. Nu se poate promite
100% împotriva tuturor echipelor. Un winrate de aproximativ 50% contra unor
adversari potriviți dinamic după rating nu indică singur nivelul modelului.

## Ce am verificat și ce nu extrapolăm

- Șase teste executate: capturi ONNX contra motorului pe partide complete,
  batch-uri dinamice și valori extreme, bonusul corect pentru a doua acțiune,
  captură prin sacrificarea gărzii, evacuarea de pe border și prudența Magenta,
  Glicko-2 contra exemplului publicat. Rezultate în `results/unit_tests.txt`.
- Contractul, resursele și meciul Level 3 au trecut validatorul oficial;
  detaliile măsurate sunt în `results/validation.txt`.
- Reconstruirea ONNX-ului final este identică octet cu octet. Fișierele
  motorului primit sunt verificate prin `results/engine_hashes.json`.
- Prima iterație, bazată predominant pe creșterea valorii Trace, a dat doar
  **54/100** la verificarea independentă; nu a fost prezentată drept avantaj
  cert. Rezultatul rămâne în `results/iteration1_final_model11.json`.
- Jurnalul istoric are 27 protocoale complete, toate rejucate: 12 victorii și
  15 înfrângeri după culoarea utilizatorului; trei jocuri fără protocol sunt
  excluse. În export, `score=1` coincide cu victoria Cyan. Nu comparăm direct
  acel 44.4% cu noul winrate local: adversarii și versiunile pot fi diferite.

## Reproducere

Din directorul proiectului, după instalarea din README:

```bash
python main.py
python -m unittest test_model -v
python tools/validate_submission.py model11_optimized.onnx
python arena.py --a model11_optimized.onnx --b baselines/model11.onnx \\
  --pairs 50 --seed 170000 --workers 4 --out results/reproduced_final.json
```

Instrucțiunile cloud `install_script` și `start_skill` sunt salvate în draft.
Pentru păstrarea configurației cloud, revizuiește și salvează setările mediului,
apoi publică mediul. Salvarea draftului nu publică și nu demonstrează restaurarea
într-o sarcină nouă. Fișierele ONNX și arhiva pot fi folosite separat de publicare.
"""
    (ROOT / "REPORT.md").write_text(body)
    (ROOT / "results/deliverable_summary.json").write_text(
        json.dumps(
            {
                "model_sha256": sha,
                "model_bytes": model.stat().st_size,
                "configuration": cfg,
                "primary": primary,
                "secondary": {name: data["summary"] for name, data in reports[1:]},
                "validation": "ACCEPTED",
                "unit_tests_passed": 6,
            },
            indent=2,
        )
        + "\n"
    )
    print(body)


if __name__ == "__main__":
    main()
