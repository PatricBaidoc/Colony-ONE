# Model11 optimizat: strategie și rezultate

## Fișierele de folosit

- `model11_optimized.onnx`: **501,803 bytes**, un singur fișier
  cu toate ponderile incluse. Validatorul oficial: **ACCEPTED**.
- `main.py`: reconstruiește modelul ales din `best_config.json` și originalul
  `baselines/model11.onnx`. `build_model.py` conține optimizarea efectivă.
- `arena.py`, `metrics.py`: comparații reproductibile și Glicko-2 local.
- `results/`: rezultate brute, configurații de selecție și verificări.

SHA-256 model final: `6b71744f625f9b3adc36eab46333fb97c42ee9cb4965afbbd08faa6dc11fdaaa`.

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
| model11 | 60/100 | 60.0% | 50.2%–69.1% |
| v2 | 16/20 | 80.0% | 58.4%–91.9% |
| claude2 | 13/20 | 65.0% | 43.3%–81.9% |

Împotriva **model11 original**, pe culori:

- Cyan: **29/50**.
- Magenta: **31/50**.

Bootstrap pe perechi, 95%: **50.0%–70.0%**.
Diferență Elo descriptivă cu regularizare: **+69.7**.
Glicko-2 local: **1569.7**, RD **35.0**, volatilitate
**0.059997**. Ipoteze: o singură perioadă, modelul nou începe la
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
python arena.py --a model11_optimized.onnx --b baselines/model11.onnx \
  --pairs 50 --seed 170000 --workers 4 --out results/reproduced_final.json
```

Instrucțiunile cloud `install_script` și `start_skill` sunt salvate în draft.
Pentru păstrarea configurației cloud, revizuiește și salvează setările mediului,
apoi publică mediul. Salvarea draftului nu publică și nu demonstrează restaurarea
într-o sarcină nouă. Fișierele ONNX și arhiva pot fi folosite separat de publicare.
