# Colony: One — optimizarea model11 pentru Level 4

Punctul de plecare este `baselines/model11.onnx` din arhiva utilizatorului.
Modelele originale și motorul jocului sunt păstrate. Nu se schimbă regulile
sau MCTS-ul arbitrului pentru a obține rezultate mai bune în teste.

## Descărcare rapidă

- [Modelul ONNX final](model11_optimized.onnx?raw=true)
- [Codul complet, ONNX și rezultatele într-un ZIP](downloads/colony-one-model11-optimized.zip?raw=true)
- [Strategia și raportul de evaluare](REPORT.md)

**Export actual:** priorități noi pentru protejarea Kernelului, eliberarea
culoarelor blocate de aliați, amenințări în două acțiuni și evacuarea tuturor
pieselor înainte de firewall. La cererea utilizatorului, ultimul export a fost
construit fără teste sau meciuri noi. Hash-urile sunt în `SHA256SUMS.txt`.

Rezultatul **60/100** și verdictul **ACCEPTED** aparțin versiunii precedente,
păstrate în `baselines/model11_strategy_v1.onnx`; nu sunt rezultate pentru noul
export. Detaliile și analiza jurnalului nou sunt în [REPORT.md](REPORT.md).

Actualizarea din partida `22872` adaugă bugetul exact până la firewall, rute
alternative, coordonare între piese, finaluri cu doi Kerneluri și un rezidual
antrenat pe 984 exemple istorice. `learn_from_logs.py` reproduce antrenarea;
ponderile sunt în `learned_weights.npz` și sunt incluse în ONNX la export.
Antrenarea nu demonstrează un winrate mai bun. Nu au fost rulate teste noi.

## Instalare

Python 3.11–3.14, pe 64 de biți. Nu este necesar GPU, PyTorch sau acces la un
serviciu extern. Din directorul proiectului:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Model și evaluare

`build_model.py` pornește întotdeauna de la model11. Parametrii se pot furniza
printr-un fișier JSON; metadatele ONNX păstrează configurația folosită.

Varianta livrată folosește Magenta prudentă și Cyan cu priori mai puțin
concentrați. Magenta penalizează mutările pe câmpuri atacate și retragerea
gărzilor de lângă propriul Kernel. La ambele culori, o captură forțată de Kernel
are prioritate. Bonusul de pregătire a capturilor este mai mare la prima dintre
cele două acțiuni. Variația efectivă în deschidere vine din sampling-ul și
noise-ul arbitrului; ONNX-ul nu introduce un generator aleator propriu.

Evacuarea urmărește **următorul inel care se închide**, inclusiv după primele
restrângeri. Kernelul primește o penalizare mult mai mare dacă ar ateriza pe
inelul condamnat. O tragere sau o captură redirecționată spre gardă lasă
atacatorul pe loc și nu primește bonusul de evacuare.

```bash
python main.py
python build_model.py --config best_config.json --output model11_optimized.onnx
python tools/validate_submission.py model11_optimized.onnx
python -m unittest test_model -v
python arena.py --a model11_optimized.onnx --b baselines/model11.onnx \
  --pairs 50 --seed 170000 --workers 4 --out results/my_evaluation.json
```

Comanda `arena.py` folosește implicit **scena Level 4** din
`scenes/level4.json` și **100 de simulări per acțiune**. O pereche înseamnă două
meciuri, câte unul la fiecare culoare. Semințele și hash-urile modelelor sunt
salvate. Noise-ul de deschidere și sampling-ul sunt cele din MCTS-ul oficial.
`--trace` salvează și mutările pentru diagnostic.

**Atenție:** `colony_one/scene.json` este scena originală Level 3, păstrată
pentru validator. Scriptul original `play/model_vs_model.py` o folosește implicit;
pentru comparații Level 4 se folosește `arena.py`, care încarcă explicit scena
corectă. Tura are două acțiuni ale aceleiași echipe.

## Ce optimizează codul

- Valoarea materialului, siguranța Kernelului, urgența firewallului și
  concentrarea probabilităților pe capturi sunt parametri reproductibili.
- Atacurile din modelul nou țin cont de blocaje, liniile de tragere și salturi.
- Riscul pentru piese și Kernel poate ajusta valoarea și politica; atacurile
  fără relocare și redirecționarea spre gărzi sunt tratate separat.
- Opțional, evaluatorul recunoaște capturarea forțată a Kernelului în tura
  curentă: o gardă poate fi eliminată la prima acțiune, Kernelul la a doua.
- Politica poate evalua capturile pregătite printr-o mutare a aceleiași piese.
  Acest bonus este mai mare când mai există două acțiuni în tură. Calculul
  păstrează piesa pe vechea poziție când verifică razele ipotetice, deci poate
  subestima liniile nou deschise; arbitrul verifică mereu regulile exacte.
- `tune.py` compară configurații pe semințe de selecție; verificarea finală
  trebuie să folosească alte semințe, stabilite înainte de citirea rezultatelor.

## Metrici și limite

`arena.py` raportează winrate total și pe culori, interval Wilson 95%, bootstrap
pe perechi, diferență Elo descriptivă și un Glicko-2 local cu ipoteze explicite.
Ratingul local nu este ratingul turneului: ratingurile, RD-urile, volatilitățile
adversarilor și perioadele serverului nu sunt disponibile. Un winrate de 50%
împotriva adversarilor potriviți după rating nu înseamnă automat model slab.

Nu se poate promite 100% împotriva echipelor necunoscute. Rezultatele față de
modelele livrate reprezintă comparații locale, nu performanță măsurată pe server.

`analyze_matches.py` reconstituie partidele din `results/provided_matches.json`.
În acest jurnal, `score=1` indică victoria Cyan, nu neapărat victoria utilizatorului.
Se folosesc câștigătorul din protocol și câmpul `you`: 12 victorii din 27 de
protocoale complete; 3 partide fără protocol sunt excluse, nu presupuse câștigate.

Specificațiile oficiale sunt în `docs/`. Consultați `LICENSE` pentru licența
motorului primit; aceasta este păstrată și se aplică codului original.
