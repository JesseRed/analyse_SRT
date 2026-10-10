# Simulationsstudie: Validierung von Chunking-Methoden mit bekannter Ground Truth

Dieser Ordner enthält simulierte SRTT-Datensätze, deren latente Chunk-Struktur **bekannt** ist.
Mit ihnen lässt sich prüfen, ob eine Chunking-Methode die Struktur findet, die tatsächlich in den Daten steckt.
Bei echten Daten ist das unmöglich, weil dort die Wahrheit unbekannt ist.

Die Studie folgt dem **ADEMP-Schema** für Simulationsstudien:
Morris, T. P., White, I. R., & Crowther, M. J. (2019). Using simulation studies to evaluate statistical methods. *Statistics in Medicine*, 38(11), 2074–2102. https://doi.org/10.1002/sim.8086

Der Code liegt in [`src/simulation/`](../src/simulation/), die Konfigurationen in [`configs/simulation/`](../configs/simulation/).

---

## Kurzanleitung

```bash
# (einmalig / bei neuen echten Daten) Kalibrierung aus den echten Dateien
python -m src.simulation calibrate            # -> configs/simulation/empirical_calibration.json
python -m src.simulation fit-noise --jobs 14  # Rauschparameter per Rastersuche anpassen

# Daten erzeugen
python -m src.simulation generate --config configs/simulation/core.yaml            # Kernset (in Git)
python -m src.simulation generate --config configs/simulation/full.yaml --jobs 14  # volles Set (~10.000 Dateien)

# Realismus prüfen
python -m src.simulation realism simulations/core/A_null simulations/core/E_realistic

# Methoden laufen lassen und bewerten
python -m src.simulation run-methods --sim-dir simulations/core --results outputs/sim_core --jobs 14
python -m src.simulation evaluate    --sim-dir simulations/core --results outputs/sim_core
```

Die Erzeugung ist **bit-genau reproduzierbar**:
- Jede Datei hat einen eigenen Seed, `np.random.SeedSequence(base_seed, spawn_key=(crc32(Familie), Zelle, Replikat, Proband, Sitzung))`.
- Der Seed steht im `manifest.csv`.
- Das volle Set wird deshalb nicht versioniert, sondern bei Bedarf neu erzeugt.

---

## A – Aims (Ziele)

1. **Validität:** Findet eine Methode die wahren Chunk-Grenzen (Position und Anzahl)?
2. **Fehlalarme:** Meldet sie Chunks, wo keine sind? Das gilt insbesondere bei motorischen Störfaktoren, Lerntrends, Heterogenität und sequenzspezifischer Beschleunigung ohne Chunking.
3. **Sensitivität / Teststärke:** Ab welcher Effektstärke wird Chunking zuverlässig erkannt, und wie hängt das vom Rauschen und vom Signaltyp ab?
4. **Dynamik:** Werden Veränderungen über Blöcke und Sitzungen erkannt (Entstehen, Zusammenfassen, Aufspalten, Verschieben, Mischung)?
5. **Inferenz auf Gruppenebene:** Reicht die Methode, um Gruppenunterschiede (jung vs. alt) nachzuweisen?
6. **Robustheit:** Wie wirken sich Datenqualität, kürzere Sitzungen, Fehler und Ausreißer aus?

---

## D – Data-generating mechanisms (Datenerzeugung)

### Design (wie in den echten Daten)

| Merkmal | Wert |
|---|---|
| Blöcke pro Sitzung | 160 (80 Blue, 40 Green, 40 Yellow), in F1 120 |
| Blockreihenfolge | eine der 11 **echten** Reihenfolgen aus den Originaldaten |
| Blue / Green | `23143124` / `34213241` (für alle gleich) |
| Yellow | fester Pool aus 8 Sequenzen, ausgewogen verteilt (enthält die Blue-Sequenz!) |
| Ereignisse pro Block | immer 8; Fehler = falsche Taste (`isHit=0`), Block läuft weiter |
| Zeitauflösung | 60-Hz-Frames (1/60 s), erster Tastendruck log-normal (Median ≈ 4,1 s) |
| Dateiformat | identisch zu den Originalen (`;`, Dezimalkomma, 7 signifikante Stellen, CRLF) |
| Dateiname | `SIM_<Familie>_<Zelle>_<uid>_Sim_<Datum>_FRA_<Sitzung>_fertig.csv` (kompatibel zum Parser) |

### Generatives Modell

Für Block *b* und IKI-Position *k* = 1..7, auf der log-Skala:

```
log IKI = μ_Proband
        + Positionseffekt[k] + Tastenpaar-Effekt[Taste_k → Taste_k+1]   (motorisch, empirisch geschätzt)
        + Sequenz-Speedup[seq]                                         (sequenzspezifisches Lernen, uniform)
        + Lernen(b) + Sitzungsgewinn + Ermüdung
        + AR(1)-Blockrauschen
        + Stärke · δ          an Grenz-IKIs                             (Spike-Kanal)
        + Stärke · ℓ · a_c    innerhalb Chunk c                         (Niveau-Kanal)
        − Stärke · ω · Rampe  innerhalb Chunk                           (Beschleunigung)
        + Stärke · u_b,c      geteiltes Chunk-Rauschen                  (Korrelations-Kanal)
        + β · Surprisal_k     (optional, n-Gramm-Lerner)                (prädiktiver Kanal)
        + Post-Error-Slowing

IKI = exp(log IKI) · R,   R = Ex-Gauß-Rauschen mit Median 1 (log-normaler Kern + Exponentialanteil)
```

Danach kommen hinzu:
- Zeitstempel-Jitter. Er erzeugt die negative Lag-1-Korrelation der echten IKIs.
- Lapses: seltene lange Pausen.
- Quantisierung auf das Frame-Raster.

**Fehler**
- Die Fehlerwahrscheinlichkeit ist positionsspezifisch und empirisch geschätzt.
- Fehler häufen sich blockweise: Die Fehlerneigung pro Block ist Gamma-verteilt.
- Zu Beginn eines Chunks ist die Wahrscheinlichkeit um den Faktor 1,75 erhöht.

**Konvention:** Eine Chunk-Grenze bei *k* bedeutet, dass IKI_k (der Übergang von Item *k* zu Item *k*+1) zwischen zwei Chunks liegt. Grenz-IKIs erhalten den Spike. Niveau-, Rampen- und Korrelationskanal wirken nur auf IKIs innerhalb eines Chunks.

### Kalibrierung an echten Daten

Alle realistischen Grundwerte stammen aus den Originaldaten (400 zufällige VR_ML-Dateien, etwa 405.000 IKIs). Dazu gehören:
- Positions- und Tastenpaar-Effekte (Ridge-Regression auf Yellow)
- Fehlerraten pro Position, Post-Error-Slowing
- Verteilung des ersten Tastendrucks
- Sequenz-Speedup (netto, ohne die Tastenpaar-Effekte der Sequenz)
- Lerntrend und die echten Blockreihenfolgen

Die Rauschparameter (Ex-Gauß-Kern und -Rand, Jitter, Block-SD, AR(1)) werden per Rastersuche so gewählt, dass die Kennwerte innerhalb eines Probanden zu den echten passen:
- Block-SD
- AR(1)
- Residual-SD (normal und robust)
- linker und rechter Rand
- Lag-1-Korrelation

Gemessen wird beides mit **demselben Schätzer**. Ergebnis und Vergleich stehen in [`core/realism_report.md`](core/realism_report.md).

**Befunde aus der Kalibrierung, die für die Methoden relevant sind:**
- Yellow ist keine Zufallsbedingung, sondern ein Pool aus 8 festen Sequenzen. Eine davon ist Blue. Yellow hat ein eigenes Positionsprofil mit langsamer Mitte (Position 4 etwa +0,12 log). Wer Yellow als „Null“ subtrahiert, subtrahiert also Struktur.
- Blue ist im Mittel etwa 0,20 log schneller als Yellow, Green etwa 0,13. Das ist ein uniformer Speedup, **kein** Chunking-Signal.
- Innerhalb eines Probanden sind log-IKIs **rechtsschief** (Ex-Gauß). Der lange linke Rand der gepoolten Rohdaten entsteht durch Unterschiede zwischen Probanden.
- Aufeinanderfolgende IKIs sind negativ korreliert (≈ −0,25), weil die Zeitstempel verrauscht sind. Methoden, die auf IKI-Ähnlichkeit beruhen (Community Network), sind davon direkt betroffen.

### Szenario-Familien

| Familie | Zellen | Faktoren | Zweck |
|---|---|---|---|
| **A_null** | A1-pure, A2-motor, A3-trends, A4-hetero, A5-seqspeedup | keine Chunks; Störfaktoren schrittweise zugeschaltet | Fehlalarmrate, Kalibrierung |
| **B_static** | 54 = 6 Effektstärken × 3 Rauschniveaus × 3 Kanäle | δ ∈ {0; 0,05; 0,10; 0,20; 0,30; 0,50}; Rauschen low/mid/high (0,6×/1×/1,5× real); Kanal spike/level/corr | Sensitivität, Präzision, d50 |
| **C_dynamics** | 8 | Entstehen, Zusammenfassen (2-2-2-2→4-4→8), Aufspalten, Grenzverschiebung, Mischung 20 %/50 %, Konsolidierung, prädiktiv ohne Chunks | Erkennen von Veränderung |
| **D_groups** | 4 = 2 Effektstärken × 2 Gruppen | jung: Chunks mit 4–5 Items; alt: 2–3 Items, langsamer, mehr Fehler | Teststärke für Gruppeneffekte |
| **E_realistic** | 1 | alles heterogen: Struktur, Dynamik, Effektstärke, Green schwächer, prädiktiver Anteil | Ende-zu-Ende-Benchmark |
| **F_quality** | 4 | 120 Blöcke, 3× Fehler, 5 % Lapses, 15 % fehlende Blöcke | Robustheit |

**Strukturen in Familie B**
- Sie werden pro Proband und Sequenz aus {8, 4-4, 2-2-2-2, 3-3-2, 2-3-3, 3-5, 5-3, 2-6, 2-2-4} gezogen.
- „8“ bedeutet keine Grenze. So enthält jede Zelle auch Dateien ohne Chunks.
- Blue und Green bekommen unabhängige Strukturen. Yellow hat nie Chunks.

**Umfang**

| Tier | Dateien | Größe | In Git | Zweck |
|---|---|---|---|---|
| `core` | 592 | ~46 MB (davon 22 MB Ground Truth) | ja | Smoke-Test, Demo, Methodenentwicklung |
| `full` | 10.200 | ~300 MB | nein (per Seed reproduzierbar) | belastbare Aussagen (kleine MCSE) |

### Dateien pro Familie

```
simulations/<tier>/<Familie>/
├── data/*.csv          # Rohdaten im Originalformat
├── ground_truth.csv    # eine Zeile pro Block (s. u.)
├── manifest.csv        # eine Zeile pro Datei: alle Generierungsparameter + Seed
└── cells.csv           # Faktorstufen jeder Zelle
```

**`ground_truth.csv`, wichtigste Spalten**

| Spalte | Bedeutung |
|---|---|
| `source_stem`, `block_number`, `sequence_type`, `target` | Schlüssel; Join auf `Path(source_file).stem` |
| `structure` | wahre Partition dieses Blocks (z. B. `3-3-2`) |
| `true_boundaries` | Grenz-IKIs 1..7 (kanonisch, gleiches Format wie `chunk_boundaries`) |
| `true_cp_segments` | was ein idealer Segmentierer melden würde (Change Points 1..6) |
| `true_n_chunks` | Anzahl Chunks |
| `strength`, `effective_delta`, `effective_level`, `effective_chunk_sd` | tatsächlich wirksame Signalstärke (Dynamik!) |
| `n_errors`, `error_positions`, `n_lapses` | Fehler- und Ausreißerinformation |
| `surprisal_profile` | Surprisal des n-Gramm-Lerners je IKI (für prädiktive Methoden) |
| `included_by_extract` | ob der Block die gemeinsame Pipeline (`extract_ikis`) übersteht |
| `window_modal_boundaries` | häufigste Partition im 30-Block-Fenster (Wahrheit für fensterbasierte Methoden) |

---

## E – Estimands (Zielgrößen)

- **Block-Ebene:** die Menge der Grenz-IKIs bzw. die Partition jedes Blocks.
- **Datei-Ebene:** mittlere Anzahl Chunks und das Vorhandensein von Chunking.
- **Verlauf:** die Veränderung der Chunk-Anzahl über die Sitzungen.
- **Gruppen-Ebene:** der Unterschied in der Chunk-Anzahl zwischen jung und alt.

## M – Methods (zu prüfende Methoden)

Alle in `src/chunking/methods` registrierten Methoden. Neue Methoden werden automatisch mitgeprüft, sobald sie eine Grenz-Konvention haben. Diese wird in `METHOD_CONVENTIONS` in [`src/simulation/evaluate.py`](../src/simulation/evaluate.py) eingetragen:

| Konvention | Bedeutung von *k* in `chunk_boundaries` | Methoden |
|---|---|---|
| `iki_index` | IKI_k ist der Übergang zwischen zwei Chunks (1..7) | rational_chunking, hcrp_lm |
| `segment_cp` | Change Point zwischen IKI-Segment *k* und *k*+1 (1..6) | community_network, change_point_pelt, hsmm_chunking |

`run-methods` setzt `n_null_runs=0`. Die Kalibrierung der Methoden wird empirisch über Familie A beurteilt, nicht über deren interne Nullmodelle.

## P – Performance measures (Gütemaße)

| Maß | Definition |
|---|---|
| `f1_conv`, `recall_conv`, `precision_conv` | **primär**: konventions-bewusster Abgleich. Ein Segment-Change-Point *c* trifft die wahre Grenz-IKI *k*, wenn *c* ∈ {*k*−1, *k*}. Bei `iki_index` gilt *c* = *k* |
| `f1_strict`, `f1_tol1` | exakter Abgleich in der eigenen Konvention der Methode bzw. mit ±1 Toleranz |
| `exact`, `hamming`, `ari` | exakte Übereinstimmung, Hamming-Abstand, Adjusted Rand Index der Partition |
| `false_alarm_rate` | Anteil der Blöcke ohne wahre Grenze, in denen trotzdem eine gemeldet wird |
| `n_chunks_bias`, `n_chunks_rmse` | Abweichung der Chunk-Anzahl (in der Konvention der Methode) |
| `auc_vs_null` | schwellenfreie Trennbarkeit von Dateien mit und ohne Chunks (Mann-Whitney) |
| `d50_recall` | Effektstärke δ, bei der 50 % der Grenzen gefunden werden |
| `mean_spearman_true_vs_pred` | Korrelation zwischen wahrem und geschätztem Verlauf über die Sitzungen |
| `power` (Familie D) | Anteil der Replikate mit p < 0,05 (Welch-t) und richtigem Vorzeichen |
| `*_mcse` | Monte-Carlo-Standardfehler per Cluster-Bootstrap über Probanden |

Fensterbasierte Methoden (PELT, Rational) werden zusätzlich gegen `window_modal_boundaries` bewertet (`f1_conv_window_modal`). So werden sie nicht für Variation von Trial zu Trial bestraft, die sie prinzipbedingt nicht auflösen können.

**Ausgaben von `evaluate`:**
- `evaluation_summary.csv`: eine Zeile pro Zelle × Methode × Sequenz
- `static_power.csv`, `null_positions.csv`, `dynamics_tracking.csv`, `group_power.csv`
- `block_scores.csv.gz`: alle Block-Scores
- `evaluation_report.md`

---

## Grenzen der Simulation

- Chunking wird als **Zeitsignal** modelliert (Spike, Niveau, Korrelation, Rampe). Wer eine Methode auf einem anderen Mechanismus aufbaut, sollte prüfen, ob dieser Mechanismus hier erzeugt wird. Gegebenenfalls einen Kanal ergänzen.
- Die Positions- und Tastenpaar-Effekte stammen aus Yellow. Weil sich die Yellow-Sequenzen wiederholen, können sie selbst schwach gelernte Struktur enthalten.
- Die Streuung zwischen Probanden liegt standardmäßig bei 50 % der echten. Die echte Streuung umfasst mehrere Studiengruppen. Familie E und A4 verwenden 100 %.
- Das Kernset ist zu klein für Inferenz. Belastbare Zahlen liefert erst `full`.
