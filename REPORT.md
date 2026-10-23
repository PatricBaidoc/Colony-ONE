# Actualizare: protecție, mobilitate și evacuare

Model: `model11_optimized.onnx`, 2,010,939 bytes.
SHA-256: `28120295260eef750c9d2d4766d0b870c32fea9582bc74f16587017637bf7b7a`.

## Priorități

1. Captura câștigătoare a Kernelului advers.
2. Supraviețuirea propriului Kernel și evacuarea sa înainte de firewall.
3. Păstrarea ieșirilor spre interior; un aliat precum Ice este încurajat să
   elibereze culoarul și penalizat dacă ocupă o ieșire rară.
4. Evacuarea tuturor pieselor, nu doar mutarea lor în același inel condamnat.
   Tragerea și atacul redirecționat spre gardă lasă atacatorul pe loc.
5. Protejarea materialului, evitarea schimburilor nefavorabile și a amenințărilor
   în două acțiuni, inclusiv capturarea gărzii urmată de atacarea Kernelului.

## Extensie din partida 22872 și cele șase optimizări

- Planificare coordonată pe două acțiuni: eliberarea culoarului unui aliat,
  urmată de captură sau evacuarea Kernelului; căutarea finală rămâne în MCTS.
- Buget calculat din observație: acțiunile proprii până la firewall sunt
  `16 * (1 - urgency) + 2 * actions_left_plane`. Când salvarea consumă întreg
  bugetul, mutările opționale care nu ajută evacuarea primesc penalizare.
- Rute alternative de Kernel până la trei pași și cost de două acțiuni pentru
  eliberarea de către un aliat a unei ieșiri adiacente spre inelul sigur.
- Tratarea finalului cu doi Kerneluri: distanța de maximum două căsuțe pierde
  înainte de departajarea după centralitate; Magenta câștigă egalitatea de inel.
- Schimburi și sacrificii compensate prin valoarea țintei și posibilitatea
  recapturării; o captură câștigătoare de Kernel rămâne prioritară.
- Învățare NumPy din mutările câștigătorilor și exemple negative slabe din
  ultimele greșeli ale învinșilor. Rezidualul învățat este limitat la ±0.2.

`match-22872.txt` este câștigat de Magenta prin schimburi coordonate și finalul
Ping C3→D3→E2. Modelul nu copiază orbește o listă de mutări. Când evaluează
mutarea Kingului D3→E2, consideră D3 eliberat, astfel încât apropierea Pingului
C3→D3 urmată de captura E2 nu mai este ascunsă de vechea poziție a Kingului.

Antrenarea folosește 984 exemple: 940 mutări câștigătoare și 44 exemple negative
slabe, în 30 de epoci. Include cele 16 mutări Magenta din partida atașată, cu
pondere suplimentară. Este imitație cu evitarea unor erori, nu AlphaZero și nu
antrenare cu un profesor de căutare profundă. Nu este măsurare de winrate.
Detalii: `results/learning_summary.json`; reproducere: `python learn_from_logs.py`.

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
nu măsurare a winrate-ului nou. Reconstruirea pozițiilor pentru antrenare este
pregătire de date, nu o partidă de evaluare.

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
