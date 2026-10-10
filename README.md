# analyse_SRT – Chunking-Analysen für die Serial Reaction Time Task (SRTT)

Dieses Projekt wertet SRTT-Daten aus: 8-Item-Sequenzen und die Zeitabstände zwischen den Tastendrücken (Inter-Key-Intervals, IKIs). Dafür stehen fünf **Chunking-Methoden** zur Verfügung. Alle Methoden laufen auf denselben Daten und schreiben ihre Ergebnisse im selben Format. So lassen sie sich wissenschaftlich vergleichen.

Dazu kommt ein **Simulations- und Validierungs-Framework** ([`src/simulation/`](src/simulation/), [`simulations/`](simulations/)). Es erzeugt realistische Datensätze, deren Chunk-Struktur bekannt ist, und misst daran, wie gut jede Methode diese Struktur findet.

**Inhalt**
1. [Setup](#voraussetzungen-und-setup) · 2. [Projektstruktur](#projektstruktur) · 3. [Datensatz & Paradigma](#datensatz--paradigma) · 4. [Analysen starten](#analysen-starten) · 5. [Methoden im Detail](#methoden-im-detail) · 6. [Bekannte Einschränkungen](#bekannte-einschränkungen-und-offene-punkte) · 7. [Ausgabe](#ausgabe-was-wird-wo-geschrieben) · 8. [Simulation & Validierung](#simulation--validierung) · 9. [Neue Methoden](#neue-chunking-methoden-hinzufügen) · 10. [Python-API](#python-api-ohne-cli) · 11. [Docker](#docker--portabilität) · 12. [Literatur](#literatur)

---

## Voraussetzungen und Setup

- **Python ≥ 3.8**. Das lokale `.venv` nutzt 3.8, das Docker-Image `python:3.11-slim`.
- `hsmm_chunking` braucht zusätzlich `pyhsmm`/`pybasicbayes` mit Patches. Am einfachsten geht das über Docker (siehe [Docker](#docker--portabilität) und [`docs/HSMM_SETUP.md`](docs/HSMM_SETUP.md)).

### Erste Einrichtung (nach Clone)

```bash
cd analyse_SRT
python3 -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate          # Windows

pip install -r requirements.txt
pip install pytest                 # optional, für die Tests
```

Ohne Aktivierung der venv:

```bash
.venv/bin/python -m src.chunking ...
```

Tests:

```bash
python -m pytest tests/
```

---

## Projektstruktur

```
analyse_SRT/
├── README.md                 # Diese Datei
├── requirements.txt
├── methods_config.json       # Standard-Parameter für alle Methoden (--config)
├── infos.md                  # Kontext: SRTT, Chunking-Theorie, Fragestellungen
│
├── algorithms/               # Ausführliche Dokumentation je Methode (Herleitung, Plan)
├── docs/HSMM_SETUP.md        # Installation des pyhsmm-Stacks
│
├── src/
│   ├── chunking/             # Analyse-Paket
│   │   ├── _data.py          # load_srt_file, extract_ikis (gemeinsame Vorverarbeitung)
│   │   ├── _base.py          # ChunkingResult, Output-Schema
│   │   ├── run.py            # Single/Batch/Benchmark-Runner
│   │   ├── output_evaluation.py, benchmark_eval.py
│   │   └── methods/          # die 5 Methoden
│   └── simulation/           # Simulation mit Ground Truth + Bewertung
│
├── configs/simulation/       # Kalibrierung (aus echten Daten) + Tier-Konfigurationen
├── simulations/              # simulierte Datensätze (core in Git, full lokal) + ADEMP-Protokoll
├── tests/                    # pytest
│
├── SRT/                      # echte Eingabedaten
├── Lit/                      # Literatur (PDFs)
└── outputs/                  # Analyseergebnisse (automatisch erstellt)
```

---

## Datensatz & Paradigma

**Dateiformat (eine CSV pro Proband und Sitzung)**
- Trennzeichen `;`, Dezimalkomma, CRLF.
- Spalten: `BlockNumber;EventNumber;Time Since Block start;isHit;target;pressed;sequence`
- `EventNumber` läuft über die ganze Datei durch. `Time Since Block start` ist in Sekunden mit 60-Hz-Frame-Auflösung angegeben.

**Blöcke**
- Jeder Block hat **immer genau 8 Tastendrücke**. Ein Fehler ist eine falsche Taste (`isHit = 0`, `pressed ≠ target`). Der Block läuft danach weiter, es gibt keine Wiederholung.
- Der erste Tastendruck kommt im Median etwa 4,1 s nach Blockstart.

**Design**
- 160 Blöcke (80 Blue / 40 Green / 40 Yellow) oder 120 Blöcke (60/30/30).
- Die Blöcke sind verschachtelt. Es gibt nur etwa 11 verschiedene feste Blockreihenfolgen.

| Sequenz | Inhalt | Rolle |
|---|---|---|
| **Blue** | `23143124`, für alle Probanden gleich | häufige Sequenz (50 %) |
| **Green** | `34213241`, für alle Probanden gleich | seltenere Sequenz (25 %) |
| **Yellow** | **fester Pool aus 8 Sequenzen**, jede etwa 5× pro Sitzung | Vergleichsbedingung (25 %) |

> **Achtung, Yellow ist keine Zufallsbedingung.** Der Pool lautet `43143142, 21413243, 13424132, 24134231, 23143124, 41324312, 12413423, 21342412`.
> - Er **enthält die Blue-Sequenz**, etwa 1/8 der Yellow-Blöcke sind also Blue.
> - Jede Pool-Sequenz kommt mehrfach vor und kann gelernt werden.
> - Yellow hat ein eigenes Positionsprofil mit langsamer Mitte.
>
> Methoden, die Yellow als „Null“ abziehen (PELT, HSMM), ziehen also auch Struktur ab.

**Dateiname:** `<Gruppe>_<Studie>_<ID>_<Vorname>_<Nachname>_<YYYYMMDD>_FRA_<Sitzung>_fertig.csv`. Daraus werden `participant_id`, `session` und `day_index` gelesen (`run._parse_source_file_meta`).

**Gemeinsame Vorverarbeitung** (`src/chunking/_data.py: extract_ikis`), genutzt von allen IKI-basierten Methoden:
- Nur **fehlerfreie** Blöcke mit genau 8 Tastendrücken der gewählten Sequenz kommen in die Analyse.
- Daraus werden 7 IKIs berechnet.
- Ein Block wird verworfen, wenn eine IKI-Position mehr als 3 SD vom Positionsmittel abweicht.

**Empirische Kennwerte** (aus 400 Dateien, siehe [`configs/simulation/empirical_calibration.json`](configs/simulation/empirical_calibration.json))
- IKI-Median etwa 0,47 s (Blue) bzw. 0,55 s (Yellow).
- Blue ist etwa 0,20 log schneller als Yellow, Green etwa 0,13. Diese Beschleunigung ist uniform und kein Chunk-Signal.
- Etwa 0,26 Fehler pro Blue-Block, 85 % der Blöcke sind fehlerfrei.
- Post-Error-Slowing etwa +0,59 log.
- Aufeinanderfolgende IKIs sind negativ korreliert (−0,25), weil die Zeitstempel verrauscht sind.
- **Die typischen Positionsprofile von Blue und Green**, etwa die Spitzen von Green an Position 2 und 4, **lassen sich vollständig durch Positions- und Tastenpaar-Effekte erklären.** Sie sind also für sich genommen kein Beleg für Chunk-Grenzen (siehe [`simulations/core/realism_report.md`](simulations/core/realism_report.md)).

---

## Analysen starten

Alle Aufrufe laufen über das Chunking-Modul. Jede Ausgabe landet in einem **eigenen Unterordner** unter `--output-dir`.

### 1. Eine Methode, eine einzelne Datei

```bash
python -m src.chunking \
  --method community_network \
  --input-file SRT/Beispiel_Teilnehmer.csv \
  --output-dir outputs/mein_lauf          # bei --input-file Pflicht (sonst Absturz)
```

Ergebnisse: `outputs/mein_lauf/community_network/blue/` (siehe Abschnitt „Ausgabe“).

### 2. Eine Methode, alle Dateien (Batch)

```bash
python -m src.chunking \
  --method rational_chunking \
  --input-dir SRT \
  --output-dir outputs/batch \
  --pattern "*.csv" \
  --sequence-type all \
  --merge-summaries \
  --limit 10
```

- `--input-dir`: Ordner mit den CSV-Dateien (nicht rekursiv)
- `--pattern`: Glob für Dateinamen (Standard: `*.csv`)
- `--sequence-type`: `blue`, `green`, `yellow` oder `all` (alle drei nacheinander, je eigener Ordner)
- `--merge-summaries`: bei `all` zusätzlich `combined_summary.csv` mit Deltas gegen Yellow
- `--limit`: höchstens so viele Dateien (zum Testen)
- `--n-null-runs`: Anzahl Nullmodell-Läufe (Standard **100**, sehr teuer bei PELT/HSMM; `0` schaltet ab)
- `--config methods_config.json`: Parameter aus der Datei überschreiben die CLI-Werte

Die methodenspezifischen Flags stehen bei der jeweiligen Methode unter [Methoden im Detail](#methoden-im-detail).

### 3. Alle Methoden gleichzeitig (Benchmark)

```bash
python -m src.chunking --method all --config methods_config.json --input-dir SRT --n-null-runs 0
```

- `benchmark_summary.csv` enthält eine Zeile pro Datei × Methode.
- **Wichtig:** Ohne `--config` werden bei `--method all` **alle methodenspezifischen Flags ignoriert**, auch `--n-null-runs`. Dann gelten die Defaults der Funktionen, inklusive `n_null_runs = 100`.

### 4. Statistische Validierung (Nullmodelle)

Jede Methode hat eine Funktion `statistical_validation`. Sie vergleicht eine Kennzahl der echten Daten mit derselben Kennzahl nach Permutation. Was permutiert wird, steht pro Methode unten. Die Ergebnisse (`p_value`, `z_score`, Nullmittel) erscheinen in `summary.csv` und unter `artifacts/<datei>/validation.json`.

```bash
python -m src.chunking --method community_network --input-file SRT/Beispiel.csv \
  --n-null-runs 100 --output-dir outputs/validation_run
```

> Die p-Werte dieser internen Nullmodelle sind **nicht automatisch kalibriert** (siehe [Einschränkungen](#bekannte-einschränkungen-und-offene-punkte)). Die echte Fehlalarmrate jeder Methode wird deshalb empirisch über die Simulationsfamilie A gemessen.

---

## Methoden im Detail

### Überblick

| | Community Network | Change Point (PELT) | HSMM | HCRP-LM | Rational Chunking |
|---|---|---|---|---|---|
| **Signal** | Ähnlichkeit benachbarter IKIs | Median-log-IKI-Profil (Ziel − Yellow) | log-IKI-Folge (− Yellow-Median) | **nur Stimulus-Reihenfolge** (keine Zeiten) | log-IKIs |
| **Auflösung** | pro Block (Multilayer über alle Blöcke) | pro Fenster (30 Blöcke, Schritt 10) | pro Block | pro Block | pro Fenster (30/10) |
| **Grenz-Konvention** | Segment-Change-Point 1–6 | Segment-Change-Point 1–6 | Segment-Change-Point 1–6 | Grenz-IKI 1–7 | Grenz-IKI 1–7 |
| **Nullmodell** | IKIs im Block mischen → Modularität Q | Ziel-IKIs mischen → `mean_n_chunks` | IKIs mischen → Log-Likelihood | Tastenfolge mischen → −Surprisal | IKIs mischen → MAP-Log-Wahrscheinlichkeit |
| **Laufzeit/Datei** (ohne Nullmodell) | ~5 s | ~6 s | ~45 s | ~1 s | < 0,1 s |
| **Abhängigkeiten** | `leidenalg`, `python-igraph` | `ruptures` | `pyhsmm` (gepatcht) | – | – |

**Grenz-Konventionen.** In `chunk_boundaries` bedeutet die Zahl *k* je nach Methode etwas anderes:
- **Grenz-IKI** (Rational, HCRP): IKI_k, also der Übergang von Item *k* zu *k*+1, liegt zwischen zwei Chunks. Wertebereich 1–7.
- **Segment-Change-Point** (CN, PELT, HSMM): Das IKI-Profil wird zwischen IKI *k* und *k*+1 geteilt. Wertebereich 1–6.

Eine einzelne langsame Grenz-IKI_k erscheint bei Segment-Methoden als **zwei** Change Points {k−1, k}. Ihr `n_chunks` ist dann etwa 2·(Grenzen)+1 statt (Grenzen)+1. Methodenvergleiche (z. B. ARI zwischen Methoden) sind deshalb nur nach Umrechnung sinnvoll. [`src/simulation/structures.py`](src/simulation/structures.py) enthält die Umrechnungen, [`src/simulation/evaluate.py`](src/simulation/evaluate.py) den konventions-bewussten Abgleich.

---

### 1. Community Network (`community_network`)

**Prinzip** (Wymbs et al. 2012; Mucha et al. 2010)
- Jeder Block wird ein Kettengraph mit 7 Knoten (IKIs).
- Benachbarte IKIs sind mit dem Gewicht `(d_max − |IKI_i − IKI_i+1|) / d_max` verbunden. `d_max` ist die größte paarweise Differenz im Block.
- Alle Blöcke werden zu einem Multilayer-Netz gekoppelt (Kopplung *C*). Darauf läuft eine Community-Detektion (Leiden, `RBConfigurationVertexPartition`, Auflösung γ).
- Der beste von `n_iter` Läufen (höchstes Q) wird verwendet.
- Chunk-Grenze = Wechsel des Community-Labels zwischen benachbarten IKIs.

| Parameter | Default (Config) | CLI | Bedeutung |
|---|---|---|---|
| `gamma` | 0,9 | `--gamma` | Auflösung; höher = mehr, kleinere Communities |
| `coupling` | 0,03 | `--coupling` | Kopplung zwischen Blöcken |
| `n_iter` | 20 (Funktion: 100) | `--n-iter` | Leiden-Wiederholungen, bester Lauf zählt |

**Outputs**
- `summary.csv`: `mean_n_chunks`, `mean_q_single_trial`, `mean_phi`, `mean_phi_normalized`, `empirical_q`, `null_q_mean/std`, `p_value`, `z_score`
- `trials.csv`: `community_labels`, `q_single_trial`, `phi` (= 1/Q), `ikis`

**Nullmodell:** IKIs innerhalb jedes Blocks werden gemischt, dann die multilayer-Modularität verglichen.

**Einschränkungen**
- `n_chunks` zählt *verschiedene Labels*, nicht Segmente. Bei Labels wie [0,0,1,1,0,0,0] ist `n_chunks` ≠ Grenzen+1.
- Die Normierung pro Block mit `d_max` macht die Methode skaleninvariant. Auch reines Rauschen ergibt etwa 2 Communities pro Block.
- Weitere Punkte stehen unter [Einschränkungen](#bekannte-einschränkungen-und-offene-punkte).

Doku: [`algorithms/community_network.md`](algorithms/community_network.md)

---

### 2. Change-Point-Detektion (`change_point_pelt`)

**Prinzip** (Killick et al. 2012; Truong et al. 2020)
1. Die Blöcke werden in gleitende Fenster aufgeteilt (`window_size` Blöcke, Schritt `step`).
2. Pro Fenster: Median-log-IKI-Profil (7 Werte) der Zielsequenz **minus** Yellow-Profil im selben Fenster.
3. PELT (`ruptures`, Kosten `l2` = stückweise konstanter Mittelwert, oder `rbf`) segmentiert dieses Profil.
4. Die Penalty wird **an Yellow-gegen-Yellow-Halbierungen kalibriert**: der kleinste Wert, bei dem im Null höchstens `fpr_target` der Profile überhaupt einen Change Point haben und im Mittel höchstens `mean_cp_target` Change Points auftreten.
5. Ein Bootstrap über die Blöcke liefert Grenzwahrscheinlichkeiten.
6. Wenn die Kalibrierung scheitert, wird automatisch mit kleineren Fenstern wiederholt.

| Parameter | Default | CLI | Bedeutung |
|---|---|---|---|
| `window_size` / `step` | 30 / 10 | `--window-size`, `--step` | Fenstergröße und Schrittweite in Blöcken |
| `min_blocks` | 6 | `--min-blocks` | Mindestzahl Blöcke je Bedingung und Fenster |
| `n_bootstrap` | 500 | `--n-bootstrap` | Bootstrap-Stichproben je Fenster |
| `fpr_target` / `mean_cp_target` | 0,05 / 0,1 | `--fpr-target`, `--mean-cp-target` | Kriterien der Penalty-Kalibrierung |
| `n_null_splits` | 32 | `--n-null-splits` | Yellow-Halbierungen für die Kalibrierung |
| `penalty_grid_size` | 80 | `--penalty-grid-size` | Kandidaten der Penalty |
| `cost_model` | `l2` | `--cost-model l2\|rbf` (`--rbf-gamma`) | Kostenfunktion |
| Presets | – | `--penalty-sensitive` (0,10/0,2), `--short-session` (20/5/4) | weniger konservativ bzw. kurze Sitzungen |
| Fallback | aus | `--allow-fallback-penalty`, `--fallback-penalty` | feste Penalty, falls die Kalibrierung unmöglich ist |

**Outputs**
- `summary.csv`: `mean_n_chunks`, `mean_bootstrap_n_chunks`, `mean_boundary_entropy`, `mean_drift`, `reliable`, `n_windows_valid`, `penalty_fallback_used`
- `trials.csv`: Grenzen des nächstgelegenen Fensters
- `validation.json`: Fenster, Heatmap der Grenzwahrscheinlichkeiten

**Nullmodell:** Die Ziel-IKIs werden gemischt (Yellow bleibt unverändert), dann der gesamte Lauf wiederholt. Kennzahl ist `mean_n_chunks`.

Doku: [`algorithms/change_point_pelt.md`](algorithms/change_point_pelt.md)

---

### 3. Hidden Semi-Markov Model (`hsmm_chunking`)

**Prinzip** (Johnson & Willsky 2013; Yu 2010)
- Ein bayessches HDP-HSMM (`pyhsmm`, Weak-Limit-Approximation, Gibbs-Sampling) mit Gauß-Emissionen auf log-IKIs.
- Für Blue/Green wird standardmäßig der Yellow-Median pro Position abgezogen.
- Die Zustandsdauern sind Poisson-verteilt.
- Jeder Block ist eine eigene Sequenz, Segmente können also keine Blockgrenzen überschreiten.
- Chunk-Grenze = Zustandswechsel innerhalb eines Blocks.
- Fenster dienen nur der Zusammenfassung (Heatmap, Drift).

| Parameter | Default | CLI | Bedeutung |
|---|---|---|---|
| `n_states` | 3 (Blue/Green), 2 (Yellow) | `--n-states` | maximale Zustandszahl (Weak Limit) |
| `max_duration` | 7 | `--max-duration` | maximale Dauer; bestimmt auch den Duration-Prior |
| `baseline_correction` | an | `--no-baseline-correction` | Yellow-Median abziehen |
| `n_gibbs_iter` | 150 | `--n-gibbs-iter` | Gibbs-Iterationen |
| `window_size`/`step`/`min_blocks` | 30/10/6 | wie PELT | nur für die Zusammenfassung |

**Outputs:** `trials.csv` mit `chunk_boundaries` je Block. `summary.csv` mit `mean_n_chunks`, `mean_drift`, `reliable` und `log_likelihood`.

**Nullmodell:** IKIs im Block mischen, neu fitten. Kennzahl ist die Log-Likelihood.

**Installation:** Docker oder [`docs/HSMM_SETUP.md`](docs/HSMM_SETUP.md). Die Methode patcht beim Import einen Fehler im pyhsmm-Vorwärts-Sampler.

Doku: [`algorithms/hsmm_chunking.md`](algorithms/hsmm_chunking.md)

---

### 4. Hierarchical CRP Language Model (`hcrp_lm`)

**Prinzip** (Éltető et al. 2022; Teh 2006)
- Ein distanzabhängiges hierarchisches Chinese-Restaurant-Process-Sprachmodell lernt online die n-Gramm-Wahrscheinlichkeiten der **Stimulusfolge** (Spalte `target`), mit Back-off auf kürzere Kontexte.
- Pro Block wird die Überraschung (Surprisal) jeder Position berechnet und innerhalb des Blocks z-standardisiert.
- Grenze bei *z* > `threshold_z`.
- Das HCRP-Modell ist direkt im Code enthalten, übernommen aus `ddHCRP_LM.py` von Éltető.

| Parameter | Default | CLI | Bedeutung |
|---|---|---|---|
| `n_levels` | 3 | `--hcrp-n-levels` | Kontexttiefe (n_levels − 1 vorherige Stimuli) |
| `strength` | 0,5 | `--hcrp-strength` | Stärke α je Ebene |
| `decay_constant` | 50 | `--hcrp-decay` (0 = kein Zerfall) | Vergessensrate λ |
| `n_samples` | 5 | `--hcrp-n-samples` | Stichproben der Sitzordnung |
| `threshold_z` | 1,0 | `--hcrp-threshold-z` | Schwelle für Grenzen |

**Outputs**
- `summary.csv`: `mean_n_chunks`, `mean_surprisal`, `mean_boundary_surprisal`
- `trials.csv`: `surprisal_profile` (7 Werte)

**Nullmodell:** Die Tastenfolge im Block wird gemischt. Kennzahl ist −mittleres Surprisal.

> **Wichtige Einschränkung.** Die Implementierung nutzt **keine Reaktionszeiten** und hat **feste Parameter**. Das Ergebnis hängt deshalb nur von der Stimulusfolge ab, und die ist für Blue und Green bei allen Probanden gleich. Alle Probanden bekommen also praktisch dieselbe Struktur (etwa 3 Chunks), egal wie sie getippt haben. Auch das hochsignifikante Nullmodell (z ≈ 40) ist trivial: Eine feste Sequenz ist vorhersagbarer als eine gemischte.
>
> Im Original (Éltető et al. 2022) werden die Modellparameter **pro Proband an die Reaktionszeiten angepasst**. Dabei bildet ein lineares Antwortmodell die Vorhersagewahrscheinlichkeiten auf RTs ab. Erst damit wird die Methode zu einem Maß für das Verhalten der Probanden.

Doku: [`algorithms/hcrp_lm.md`](algorithms/hcrp_lm.md)

---

### 5. Rational Chunking (`rational_chunking`)

**Prinzip** (Wu et al. 2023)
- Pro Fenster werden alle 2⁷ = 128 möglichen Grenzmuster bewertet:
  `J(R) = NLL(log-IKIs | R) + κ·Fehlerkosten + λ·(Anzahl Grenzen)`
- Das Zeitmodell: log-IKI ~ N(μ + δ·[Grenze], σ²). Dabei ist μ der Fenstermittelwert und σ die Fenster-SD. δ wird heuristisch geschätzt: das obere 20 %-Mittel minus das restliche Mittel, mit Fallback 0,5.
- Eine Softmax mit inverser Temperatur β ergibt eine Posterior über die Partitionen.
- Ausgaben: MAP-Partition, Grenzwahrscheinlichkeiten und erwartete Chunk-Zahl.

| Parameter | Default | CLI | Bedeutung |
|---|---|---|---|
| `lam` | 1,0 | `--lam` | Komplexitätskosten pro Grenze |
| `kappa` | 1,0 | `--kappa` | Gewicht der Fehlerkosten |
| `beta` | 5,0 | `--beta` | Rationalität (inverse Temperatur) |
| `window_size`/`step`/`min_blocks` | 30/10/6 | `--window-size`, `--step`, `--min-blocks` | Fenster |

**Outputs**
- `summary.csv`: `mean_n_chunks` (Erwartungswert), `avg_boundary_probs` (7 Werte), `map_log_prob`
- `trials.csv`: MAP-Grenzen je Block (vom nächstgelegenen Fenster)

**Nullmodell:** IKIs im Block mischen. Kennzahl ist die MAP-Log-Wahrscheinlichkeit.

Doku: [`algorithms/rational_chunking.md`](algorithms/rational_chunking.md)

---

## Bekannte Einschränkungen und offene Punkte

Diese Punkte sind dokumentiert, aber **noch nicht behoben**. Sie wurden bei der Code-Durchsicht und mit den Simulationsdaten gefunden. Die Auswirkungen lassen sich mit dem [Simulations-Framework](#simulation--validierung) messen.

| Methode | Punkt | Folge |
|---|---|---|
| Community Network | `mean_phi_normalized` ist der Mittelwert von (φ − φ̄)/φ̄, also immer 0 | Kennzahl ohne Information |
| Community Network | Echte Daten bekommen den besten von `n_iter` Leiden-Läufen, das Nullmodell nur 1 Lauf | Q echt systematisch höher, p-Werte zu liberal |
| Community Network | Normierung pro Block mit `d_max` | auch reines Rauschen ergibt etwa 2 Communities |
| Community Network | `methods_config.json` enthält noch den veralteten Schlüssel `n_permutations` | `--method community_network --config methods_config.json` bricht ab |
| PELT | Die Penalty wird an Yellow-Halbierungen mit nur etwa 3–4 Blöcken kalibriert, das Signal nutzt etwa 15 Blue-Blöcke | Penalty zu hoch, fast immer 1 Chunk |
| PELT | Im Nullmodell bleibt das Yellow-Profil erhalten (gemischtes Ziel − ungemischtes Yellow) | Nullmodell nicht strukturfrei |
| PELT | Bei 120 Blöcken haben 30er-Fenster oft weniger als 6 Yellow-Blöcke | viele ungültige Fenster, Retry mit kleineren Fenstern |
| PELT, HSMM | Yellow als Baseline enthält selbst Struktur (Pool inklusive Blue-Sequenz) | Baseline nicht neutral |
| HSMM | Duration-Prior Gamma(α₀ = 2·max_duration = 14, β₀ = 2) hat seinen Mittelwert bei 7 = Sequenzlänge | begünstigt 1 Chunk |
| HSMM | Mischen innerhalb des Blocks lässt die Emissionsverteilung unverändert | schwaches Nullmodell |
| Rational | Der `kappa`-Term ist für alle Partitionen gleich | Fehler beeinflussen das Ergebnis nicht |
| Rational | μ enthält die Grenz-IKIs, δ kommt aus einer Heuristik (Fallback 0,5) | erst große Effekte erkennbar |
| HCRP-LM | nutzt keine RTs, feste Parameter (s. o.) | kein Maß für das Verhalten der Probanden |
| CLI | `--input-file` ohne `--output-dir` | Absturz (`Path(None)`) |
| CLI | `--method all` ohne `--config` | alle methodenspezifischen Flags ignoriert, `n_null_runs = 100` |
| Konventionen | Grenz-IKI (1–7) und Segment-CP (1–6) stehen in derselben Spalte | Methodenvergleiche nur nach Umrechnung |

---

## Ausgabe: Was wird wo geschrieben?

Jeder Lauf schreibt in **einen** Ordner mit festem Schema:

```
outputs/<dein_output_dir>/<method_name>/<sequence>/
├── meta.json          # Lauf-Infos: Methode, Zeitstempel, Git-Commit, n_files, …
├── parameters.json    # alle Laufparameter
├── summary.csv        # eine Zeile pro Quelldatei
├── trials.csv         # eine Zeile pro Block
├── errors.csv         # fehlgeschlagene Dateien (source_file, error)
├── progress.log       # Fortschritt (Batch)
└── artifacts/<datei>/ # trials.csv + validation.json je Quelldatei
```

### summary.csv

- **Mindestspalten:** `source_file`, `sequence_type`, `method`, `n_blocks`
- **Kern-Metrik:** `mean_n_chunks` (mittlere Chunk-Anzahl pro Block; bei Rational ein Erwartungswert)
- **Metadaten:** `participant_id`, `session`, `day_index` (aus dem Dateinamen)
- **Methodenspezifisch:** siehe die jeweilige Methode

### trials.csv

- **Mindestspalten:** `source_file`, `sequence_type`, `method`, `block_number`, `n_chunks`, `chunk_boundaries`
- `chunk_boundaries` ist eine Liste von Grenzpositionen. **Die Bedeutung hängt von der Konvention der Methode ab** (siehe [Überblick](#überblick)).
- **Metadaten:** `participant_id`, `session`, `day_index`

### meta.json / parameters.json

Dienen der Reproduzierbarkeit: Zeitstempel, Git-Commit, Eingabe-Pfad/Pattern und alle Parameter.

### combined_summary.csv (bei `--sequence-type all --merge-summaries`)

Führt `blue/`, `green/` und `yellow/summary.csv` zusammen, inklusive Delta-Spalten gegen Yellow.

### Auswerteroutine (Chunking über Tage und Sequenzen)

```bash
python -m src.chunking.output_evaluation outputs/<lauf>/change_point_pelt --csv outputs/<lauf>/change_point_pelt/evaluation_data.csv
```

Erzeugt `evaluation_report.md`: den Verlauf über die Tage pro Proband und Sequenz sowie Blue/Green gegen Yellow.

### Benchmark-Auswertung (Vergleich mehrerer Methoden)

```python
from src.chunking import benchmark_eval

df = benchmark_eval.load_benchmark_summary("outputs/benchmark")     # Datei × Methode
tab = benchmark_eval.comparison_table("outputs/benchmark")
ari_df = benchmark_eval.ari_per_file("outputs/benchmark",
                                     method_a="community_network", method_b="rational_chunking")
```

> `ari_per_file` vergleicht die rohen `chunk_boundaries`. Bei Methoden mit unterschiedlicher Konvention muss vorher umgerechnet werden (siehe [Überblick](#überblick)).

---

## Simulation & Validierung

Auf echten Daten lässt sich nicht prüfen, ob eine Methode „richtig“ liegt, weil die wahre Chunk-Struktur unbekannt ist. Das Simulations-Framework erzeugt deshalb Datensätze **im exakten Originalformat** mit **bekannter** Chunk-Struktur und bewertet jede Methode daran.

**Studienprotokoll** (ADEMP: Ziele, Datenerzeugung, Zielgrößen, Methoden, Gütemaße): [`simulations/README.md`](simulations/README.md)

**Generatives Modell, kurz**
- log-IKI = Probandenniveau + Positions- und Tastenpaar-Effekte + sequenzspezifischer Speedup + Lern- und Ermüdungstrend + AR(1)-Blockrauschen + Chunk-Signal.
- Das Chunk-Signal hat vier getrennt schaltbare Kanäle: Grenz-Spike δ, Niveau, Beschleunigung innerhalb des Chunks, geteiltes Rauschen innerhalb des Chunks.
- Dazu kommen ein optionaler prädiktiver Anteil (Surprisal), multiplikatives Ex-Gauß-Rauschen, Zeitstempel-Jitter, Lapses, 60-Hz-Quantisierung und Fehler. Die Fehler häufen sich blockweise und sind an Chunk-Anfängen häufiger.

**Kalibrierung**
- Alle Grundwerte werden aus den echten Daten geschätzt (`calibrate`).
- Die Rauschparameter werden per Rastersuche an die Kennwerte innerhalb eines Probanden angepasst (`fit-noise`).
- Ergebnis: [`simulations/core/realism_report.md`](simulations/core/realism_report.md).

| Familie | Inhalt | misst |
|---|---|---|
| A_null | keine Chunks; Störfaktoren schrittweise | Fehlalarmrate |
| B_static | 6 Effektstärken × 3 Rauschniveaus × 3 Signal-Kanäle (54 Zellen) | Sensitivität, d50 |
| C_dynamics | Entstehen, Zusammenfassen, Aufspalten, Verschieben, Mischung, Konsolidierung, prädiktiv | Erkennen von Veränderung |
| D_groups | jung (4–5 Items) vs. alt (2–3 Items) | Teststärke für Gruppen |
| E_realistic | alles heterogen | Ende-zu-Ende |
| F_quality | 120 Blöcke, viele Fehler, Lapses, fehlende Blöcke | Robustheit |

```bash
python -m src.simulation generate --config configs/simulation/core.yaml        # Kernset (bereits in Git)
python -m src.simulation generate --config configs/simulation/full.yaml --jobs 14
python -m src.simulation run-methods --sim-dir simulations/core --results outputs/sim_core --jobs 14
python -m src.simulation evaluate    --sim-dir simulations/core --results outputs/sim_core
```

- **Kernset** `simulations/core/`: 592 Dateien, in Git, für Entwicklung und Smoke-Tests.
- **Volles Set** `simulations/full/`: 10.200 Dateien. Es ist nicht in Git, aber per Seed bit-genau reproduzierbar.
- Die Ergebnisse auf dem Kernset (erste Bewertung der 5 Methoden) stehen in [`simulations/core/baseline_evaluation.md`](simulations/core/baseline_evaluation.md).

---

## Neue Chunking-Methoden hinzufügen

1. **Implementierungsvorschlag** als MD-Datei anlegen, z. B. `algorithms/meine_methode.md` (Ziel, Schritte, Literatur).
2. **Implementierungsvoraussetzungen** durchgehen: [`algorithms/Implementierungsvoraussetzungen.md`](algorithms/Implementierungsvoraussetzungen.md).
3. **Implementierung** in `src/chunking/methods/<name>.py` mit
   `run_analysis(filepath, sequence_type, **kwargs) -> ChunkingResult`.
   Die IKIs bitte über `extract_ikis` holen, damit alle Methoden dieselben Blöcke sehen.
4. **Registrierung** in `src/chunking/methods/__init__.py`:
   - den Namen in `BUILTIN_METHODS` und einen Import-Zweig in `get_runner` ergänzen,
   - oder den Decorator `@register("<name>")` nutzen.
5. **Grenz-Konvention festlegen und eintragen**: `METHOD_CONVENTIONS` in [`src/simulation/evaluate.py`](src/simulation/evaluate.py), Wert `iki_index` oder `segment_cp`. Fensterbasierte Methoden zusätzlich in `WINDOW_METHODS`.
6. **Validieren:**
   ```bash
   python -m src.simulation run-methods --sim-dir simulations/core --results outputs/sim_core --methods <name>
   python -m src.simulation evaluate --sim-dir simulations/core --results outputs/sim_core --methods <name>
   ```
   Für belastbare Zahlen danach auf `simulations/full` wiederholen.
7. **Dokumentation:** Algorithmus-MD und ein Abschnitt unter [Methoden im Detail](#methoden-im-detail).

In [`algorithms/Implementierungsvoraussetzungen.md`](algorithms/Implementierungsvoraussetzungen.md), Abschnitt 8, steht eine **Prompt-Vorlage** für eine Implementierung per KI-Assistent.

---

## Python-API (ohne CLI)

```python
from pathlib import Path
from src.chunking import run_single_file, run_batch, run_benchmark, load_srt_file, extract_ikis, list_methods

result = run_single_file("community_network", Path("SRT/Beispiel.csv"), Path("outputs/api_test"),
                         sequence_type="blue", gamma=0.9, coupling=0.03, n_null_runs=0)

run_batch(method_name="rational_chunking", input_dir="SRT", output_dir="outputs/batch",
          sequence_type="blue", limit=5, random_state=42, n_null_runs=0)

# Simulation
from src.simulation.generate import generate_from_file
from src.simulation.evaluate import run_evaluation, score_block
score_block(pred=[3, 4], true_boundaries=[4], convention="segment_cp")   # -> f1_conv = 1.0
```

Notebooks können weiterhin `run_full_analysis` aus `src.chunking` verwenden (Legacy-API, Community Network).

---

## Kurzüberblick

| Was du tun willst | Befehl / Ort |
|---|---|
| Eine Datei analysieren | `python -m src.chunking --method <m> --input-file <pfad> --output-dir outputs/x` |
| Viele Dateien, eine Methode | `--input-dir SRT --output-dir outputs/batch --sequence-type all` |
| Alle Methoden vergleichen | `--method all --config methods_config.json --n-null-runs 0` |
| Methoden verstehen | [Methoden im Detail](#methoden-im-detail), `algorithms/*.md` |
| Methode an Ground Truth prüfen | `python -m src.simulation run-methods …` + `evaluate …` |
| Simulationsdesign verstehen | [`simulations/README.md`](simulations/README.md) |
| Neue Methode einbauen | [Neue Chunking-Methoden](#neue-chunking-methoden-hinzufügen) |

---

## Docker & Portabilität

Das Projekt ist vollständig dockerisiert. So ist die Umgebung einheitlich, und die nötigen Patches für die Legacy-Abhängigkeiten (`pyhsmm`, `pybasicbayes`) werden automatisch angewendet. Image: `python:3.11-slim`.

### 1. Einrichtung mit Docker
```bash
docker build -t analyse-srt .
```
Voraussetzung: Die Ordner `pyhsmm/` und `pybasicbayes/` liegen lokal vor. Sie sind nicht in Git, siehe [`docs/HSMM_SETUP.md`](docs/HSMM_SETUP.md).

### 2. Ausführung
```bash
docker compose run analyse-srt --method hcrp_lm --limit 5
```
*Die lokalen Ordner `SRT` und `outputs` werden automatisch in den Container gemountet.*

### 3. Transfer auf einen anderen Rechner

**A. Quellcode-Transfer (empfohlen für Entwicklung)**
1. Den gesamten Projektordner kopieren, inklusive `Dockerfile`, `pyhsmm/` und `pybasicbayes/`.
2. Auf dem Zielrechner `docker build -t analyse-srt .` ausführen.

**B. Image-Export (Schnellstart ohne Build)**
```bash
docker save analyse-srt | gzip > analyse-srt-image.tar.gz   # Quellrechner
docker load < analyse-srt-image.tar.gz                      # Zielrechner
```

### 4. Troubleshooting
Falls in der Umgebung (z. B. WSL2) die `docker compose`-Plugins fehlen:
```bash
docker run --rm \
  -v $(pwd)/SRT:/app/input \
  -v $(pwd)/outputs:/app/outputs \
  analyse-srt \
  --method hcrp_lm --input-dir /app/input --output-dir /app/outputs/run1
```

---

## Literatur

**Community Network**
- Wymbs, N. F., Bassett, D. S., Mucha, P. J., Porter, M. A., & Grafton, S. T. (2012). Differential recruitment of the sensorimotor putamen and frontoparietal cortex during motor chunking in humans. *Neuron, 74*(5), 936–946.
- Mucha, P. J., Richardson, T., Macon, K., Porter, M. A., & Onnela, J.-P. (2010). Community structure in time-dependent, multiscale, and multiplex networks. *Science, 328*(5980), 876–878.
- Traag, V. A., Waltman, L., & van Eck, N. J. (2019). From Louvain to Leiden: guaranteeing well-connected communities. *Scientific Reports, 9*, 5233.
- Bassett, D. S., Wymbs, N. F., Porter, M. A., Mucha, P. J., Carlson, J. M., & Grafton, S. T. (2011). Dynamic reconfiguration of human brain networks during learning. *PNAS, 108*(18), 7641–7646.

**Change-Point-Detektion**
- Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost. *JASA, 107*(500), 1590–1598. https://doi.org/10.1080/01621459.2012.737745
- Truong, C., Oudre, L., & Vayatis, N. (2020). Selective review of offline change point detection methods. *Signal Processing, 167*, 107299. https://doi.org/10.1016/j.sigpro.2019.107299

**HSMM**
- Johnson, M. J., & Willsky, A. S. (2013). Bayesian nonparametric hidden semi-Markov models. *Journal of Machine Learning Research, 14*, 673–701.
- Yu, S.-Z. (2010). Hidden semi-Markov models. *Artificial Intelligence, 174*(2), 215–243.

**HCRP-LM**
- Éltető, N., Nemeth, D., Janacsek, K., & Dayan, P. (2022). Tracking human skill learning with a hierarchical Bayesian sequence model. *PLoS Computational Biology, 18*(11), e1009866. https://doi.org/10.1371/journal.pcbi.1009866
- Teh, Y. W. (2006). A hierarchical Bayesian language model based on Pitman-Yor processes. *Proceedings of ACL 2006*, 985–992.

**Rational Chunking**
- Wu, S., Éltető, N., Dasgupta, I., & Schulz, E. (2023). Chunking as a rational solution to the speed–accuracy trade-off in a serial reaction time task. *Scientific Reports, 13*, 7680. https://doi.org/10.1038/s41598-023-31500-3
- Lai, L., Huang, A. Z. X., & Gershman, S. J. (2025). Action chunking as conditional policy compression. *Cognition, 264*, 106201. https://doi.org/10.1016/j.cognition.2025.106201

**Chunking-Signaturen und Validierung**
- Acuna, D. E., Wymbs, N. F., Reynolds, C. A., Picard, N., Turner, R. S., Strick, P. L., Grafton, S. T., & Kording, K. P. (2014). Multifaceted aspects of chunking enable robust algorithms. *Journal of Neurophysiology, 112*(8), 1849–1856.
- Morris, T. P., White, I. R., & Crowther, M. J. (2019). Using simulation studies to evaluate statistical methods. *Statistics in Medicine, 38*(11), 2074–2102. https://doi.org/10.1002/sim.8086
