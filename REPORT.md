# Actualizare: protecție, mobilitate și evacuare

Model: `model11_optimized.onnx`, 944,687 bytes.
SHA-256: `731a8b0ddbd5401f288a0833037128b1678efe92c8f5f03968eae85a116dd8bd`.

## Priorități

1. Captura câștigătoare a Kernelului advers.
2. Supraviețuirea propriului Kernel și evacuarea sa înainte de firewall.
3. Păstrarea ieșirilor spre interior; un aliat precum Ice este încurajat să
   elibereze culoarul și penalizat dacă ocupă o ieșire rară.
4. Evacuarea tuturor pieselor, nu doar mutarea lor în același inel condamnat.
   Tragerea și atacul redirecționat spre gardă lasă atacatorul pe loc.
5. Protejarea materialului, evitarea schimburilor nefavorabile și a amenințărilor
   în două acțiuni, inclusiv capturarea gărzii urmată de atacarea Kernelului.

Sunt folosite idei de șah — mobilitate, protecție, schimburi și pregătirea
atacurilor — adaptate regulilor Colony: One. Motorul de joc și MCTS-ul
arbitrului rămân neschimbate. Euristicile ghidează căutarea; nu garantează
absența tuturor greșelilor sau existența unei scăpări din orice poziție.

## Jurnalul nou, analizat fără replay

30 înregistrări, 23 protocoale complete: 8 victorii și 15 înfrângeri după culoare;
7 meciuri fără protocol sunt excluse. Observații concrete:

- `21434`: Cyan lasă K la A1, cu Ice B2 pe ieșirea interioară directă. Acțiunile
  29–30 sunt consumate pe Daemon C4→E4→E6, apoi firewallul distruge Kernelul.
  Ice nu dovedește imobilizare totală: era posibil și un ocol în două mutări.
- `21744`: Ping advers F8→G7 capturează Ice, apoi G7→H8 capturează Kernelul.
- `19618`: Daemon advers C5→F5→F3 capturează Kernelul după apropiere.
- `18883`: Trace H7→H6 rămâne în același inel și arde la runda 8.
- `18485`: Ice și Exploit sunt pierdute după secvențe apropiere Trace + tragere;
  apoi cade Kernelul rămas fără protecție.

## Starea verificării

La cererea utilizatorului, **nu au fost rulate teste, meciuri sau validatorul
oficial pentru acest ultim export**. Construirea ONNX include verificarea
structurală standard a grafului. Analiza jurnalului este citire de date,
nu simulare și nu măsurare a winrate-ului nou.

Înainte de cererea de oprire a testării, varianta intermediară cu evacuarea
Kernelului a trecut șapte teste și validatorul. Un caz construit a verificat
Ice → eliberare culoar → King spre interior înaintea rundelor 8 și 16.
Aceste rezultate sunt istorice și nu validează adăugirile finale de protecție.

Rezultatul de 60/100 aparține versiunii anterioare, disponibilă în
`baselines/model11_strategy_v1.onnx`. Raportul ei este păstrat în
[raportul v1](results/reports/v1_strategy_report.md). Nu atribuim acel winrate
exportului nou. Jurnalul primit este în `results/new_tournament_matches.json`.

## Utilizare

`python main.py` reconstruiește exportul din `best_config.json` și model11.
Codul principal este `build_model.py`. Modelele ONNX sunt fișiere complete,
cu ponderile incluse. `README.md` conține linkurile de descărcare.
