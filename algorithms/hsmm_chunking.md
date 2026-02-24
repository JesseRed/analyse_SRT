# HSMM Chunking

Implementierungsplan und Referenz zur Implementierung in `src/chunking/methods/hsmm_chunking.py`.

---

## Ziel

Chunk-Erkennung über ein **Hidden Semi-Markov Model (HSMM)** mit expliziten Zustandsdauern. Jeder 8-Item-Block wird als Beobachtungssequenz der Länge 7 (IKIs) modelliert; Chunk-Grenzen entsprechen Zustandswechseln, Chunk-Längen den Durations. Ein Modell pro Datei×Sequenz (blue/green/yellow); Fenster-Auswertung ohne Re-Fit für dynamische Metriken.

**Wichtig:** Ergebnisse werden sowohl aggregiert (`trials.csv`) als auch einzeln pro Teilnehmer unter `artifacts/<filename>/trials.csv` gespeichert, um detaillierte Längsschnitt-Analysen zu ermöglichen.

---

## 1. Datenaufbereitung

- **Gemeinsame Pipeline:** `load_srt_file` und `extract_ikis` aus `src.chunking._data`.
- Pro Block: IKIs aus „Time Since Block start“, nur Blöcke mit allen 8 Hits. Log-Transformation \(x_{pos} = \log(IKI_{pos})\).
- **Baseline-Korrektur (Variante A):** Median pro Position aus Yellow; für Blue/Green: \(y_{pos} = x_{pos} - m_{pos}^{yellow}\). Yellow bleibt unkorrigiert (Nullsignal).

---

## 2. Datenstruktur fürs Modell

- Liste von Sequenzen: eine Sequenz = ein Block = Array der Länge 7 (log-IKI oder baseline-korrigiert).
- Segmente laufen nicht über Block-Grenzen (Decoding pro Block).

---

## 3. Modell (Bayesian HDP-HSMM, pyhsmm)

- **Bibliothek:** [pyhsmm](https://github.com/mattjj/pyhsmm) (Weak-Limit HDP-HSMM, Gibbs-Sampling).
- **Zustände:** Weak-Limit-Nmax (Blue/Green: 3, Yellow: 2 als Orientierung; Nmax z.B. 6+ für HDP).
- **Max Duration:** 7 (CLI `--max-duration`); `trunc=7` pro Sequenz.
- **Emission:** 1D Gaussian pro Zustand (NIW-Prior: mu_0, sigma_0, kappa_0, nu_0).
- **Durations:** PoissonDuration (1-indexed) pro Zustand (Gamma-Prior: alpha_0, beta_0).
- **Transition:** HDP-Transitions (alpha, gamma).
- **Inferenz:** Gibbs-Sampling (`resample_model()`); **Zustandsfolge:** nach dem Sampling aus `model.states_list[i].stateseq` pro Block → Chunk-Grenzen = Positionen mit Zustandswechsel (1..6).
- **CLI:** `--n-gibbs-iter` (Default 150) für die Anzahl Gibbs-Iterationen.

---

## 4. Mapping HSMM → Output

- **State-Sequenz pro Block** (aus pyhsmm nach Gibbs-Sampling) → Zustandsfolge (Länge 7). Zustandswechsel bei Position \(i\) → Grenze bei \(i\) (1-indexed: 1..6).
- **chunk_boundaries:** Liste dieser Positionen; **n_chunks** = len(boundaries) + 1.
- **Trials:** eine Zeile pro Block mit `block_number`, `n_chunks`, `chunk_boundaries`.

---

## 5. Fenster-Auswertung (ohne Re-Fit)

- Fenster wie CPD: `window_size=30`, `step=10` (CLI).
- Pro Fenster: State-Sequenzen der Blöcke im Fenster (bereits aus dem einen Modell-Lauf); **Boundary Probability** \(p_{b,c}(pos)\) = Anteil der Blöcke mit Grenze bei \(pos\).
- **Validation:** `boundary_heatmap` (Fenster × Position), `drift`, `window_n_chunks`.
- Modell wird einmal pro Datei×Sequenz per Gibbs-Sampling geschätzt; Fenster nutzen die gleichen pro-Block-State-Sequenzen.

---

## 6. Null-Check (Yellow)

- Yellow als Null-Bedingung; Erwartung: wenig stabile Grenzen.
- **reliable:** z.B. `n_windows_valid >= min_windows_reliable` (Default 3).

---

## 7. Implementierung und Abhängigkeit

- **Code:** `src/chunking/methods/hsmm_chunking.py`
- **Abhängigkeit:** [pyhsmm](https://github.com/mattjj/pyhsmm) (Bayesian HDP-HSMM, Gibbs-Sampling). Installation:
  ```bash
  pip install pyhsmm
  ```
  Hinweis: pyhsmm ist nicht mehr aktiv gewartet; getestet bis Python 3.7. Optional: gcc mit `-std=c++11` für schnelle Eigen-State-Klasse.

---

## Literatur

- Johnson, M. J., & Willsky, A. S. (2013). Bayesian Nonparametric Hidden Semi-Markov Models. *Journal of Machine Learning Research*, 14, 673–701.
- Johnson, M. J., & Willsky, A. S. (2010). The Hierarchical Dirichlet Process Hidden Semi-Markov Model. *UAI 2010*.
- pyhsmm: mattjj/pyhsmm (GitHub).
- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. *Proceedings of the IEEE*, 77(2), 257–286.
