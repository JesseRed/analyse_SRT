# HCRP-LM Chunking

Implementierungsplan und Referenz zur Implementierung in `src/chunking/methods/hcrp_lm.py`.

---

## Ziel

Chunk-Erkennung über ein **distance-dependent Hierarchical Chinese Restaurant Process Language Model (ddHCRP-LM)**. Das Modell lernt n-Gramm-Wahrscheinlichkeiten hierarchisch: Tiefere Kontextfenster „erben" Evidenz von flacheren Fenstern (Back-off). Die Überraschung (*Surprisal*) an jeder Position einer Sequenz wirkt als Chunk-Grenz-Signal: Positionen mit hoher Überraschung markieren den Beginn eines neuen Chunks.

**Wichtig:** Ergebnisse werden sowohl aggregiert (`trials.csv`) als auch einzeln pro Teilnehmer unter `artifacts/<filename>/trials.csv` gespeichert, um detaillierte Längsschnitt-Analysen zu ermöglichen.

---

## 1. Datenaufbereitung

- **Gemeinsame Pipeline:** `load_srt_file` aus `src.chunking._data`.
- Pro Block: alle 8 Hits (wie in `extract_ikis`). `target`-Spalte = Stimulus-Identität (ganzzahlig, 1–4).
- Sequenz über alle Blöcke hintereinander als Token-Liste: `[t_1, t_2, ..., t_{8×N}]`.
- Getrennt pro `sequence_type` (blue/green/yellow).

---

## 2. HCRP-LM Modell

Basiert auf dem [ddHCRP_LM.py](https://github.com/noemielteto/HCRP_sequence_learning) von Eltető et al. (2022), vendored direkt in die Implementierung (keine externe Abhängigkeit).

**Parameter:**

| Parameter | Default | Bedeutung |
|---|---|---|
| `n_levels` | `3` | Tiefe der Hierarchie (max. Kontext = n_levels−1 vorherige Stimuli) |
| `strength` | `[0.5, 0.5, 0.5]` | Stärke-Parameter α pro Ebene |
| `decay_constant` | `[50.0, 50.0, 50.0]` | Vergessensrate λ (exponentieller Abfall; None = nicht-zeit-abhängig) |
| `n_samples` | `5` | Unabhängige Stichproben der Sitzanordnung |
| `threshold_z` | `1.0` | Z-Score-Schwellenwert für Surprisal → Chunk-Grenze |

**Modellbeschreibung:**
- Das HCRP-Modell repräsentiert Restaurants auf `n_levels` Ebenen; jede Ebene modelliert P(Stimulus | Kontext der Länge n).
- Tiefere Ebenen erben Evidenz von flacheren durch einen Back-off-Mechanismus (Hierarchical Dirichlet Process).
- Mit `decay_constant` werden ältere Beobachtungen exponentiell vergessen (distance-dependent CRP).

---

## 3. Online-Parsing

Das Modell wird **online** über die gesamte Block-Sequenz eines Teilnehmers aktualisiert:

1. Für jeden Stimulus `w_t` mit Kontext `u_t = (w_{t-n+1}, ..., w_{t-1})`:
   - **Evaluate:** Berechne `P(w_t | u_t)` → Surprisal = `–log P(...)`.
   - **Update:** Füge `w_t` mit Kontext `u_t` ins Modell ein (`add_customer`).
2. Ergebnis: pro Trial (Stimulus) ein Surprisal-Wert.

---

## 4. Chunk-Grenz-Erkennung

Pro Block (8 Stimuli, Positionen 1–8):

- Positions 2–8 haben Kontext ≥ 1 → Surprisal-Profil der Länge 7.
- **Standardisierung:** Z-Score pro Block über die 7 Werte.
- **Grenze** bei Position `p` (als IKI-Grenz-Index 1–7), wenn `z_p > threshold_z`.
- Sonderfall Block 1: zu wenig Prior → Modell liefert Uniform-Prior; Grenzen trotzdem berechnet, aber typisch keine starken Spikes.

---

## 5. Ausgabe

### summary.csv

| Spalte | Beschreibung |
|---|---|
| `source_file` | Quelldatei |
| `sequence_type` | blue/green/yellow |
| `method` | `hcrp_lm` |
| `n_blocks` | Anzahl gültiger Blöcke |
| `mean_n_chunks` | Mittlere Chunk-Anzahl pro Block |
| `mean_surprisal` | Mittlerer Surprisal-Wert (alle Positionen, alle Blöcke) |
| `mean_boundary_surprisal` | Mittlerer Surprisal-Wert an gefundenen Grenzen |

### trials.csv

| Spalte | Beschreibung |
|---|---|
| `block_number` | Block-ID |
| `n_chunks` | Anzahl Chunks im Block |
| `chunk_boundaries` | Liste von Grenzpositionen 1–7 |
| `surprisal_profile` | Surprisal-Werte an Positionen 2–8 (Länge 7) |

---

## 6. Implementierung und Abhängigkeiten

- **Code:** `src/chunking/methods/hcrp_lm.py`
- **Abhängigkeiten:** Keine neuen. Nur `numpy`, `scipy`, `pandas` (bereits in `requirements.txt`).
- **Klasse HCRP_LM:** Vendored direkt in die Implementierung, adaptiert aus [ddHCRP_LM.py](https://github.com/noemielteto/HCRP_sequence_learning/blob/main/ddHCRP_LM.py).

---

## Literatur

- Eltető, N., Sen, K., Bhaskaran-Nair, K., & Bhattacharya, B. (2022). Tracking human skill learning with a hierarchical Bayesian sequence model. *bioRxiv*. https://doi.org/10.1101/2022.01.27.477977
- Teh, Y. W. (2006). A hierarchical Bayesian language model based on Pitman-Yor processes. *ACL 2006*.
- Blei, D. M., & Frazier, P. I. (2011). Distance dependent Chinese restaurant processes. *JMLR*, 12, 2461–2488.
