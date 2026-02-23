# Change-Point-Chunking (PELT + ruptures)

Implementierungsplan. Die Implementierung erfolgt in `src/chunking/methods/change_point_pelt.py` (noch nicht umgesetzt).

---

## Ziel (klarer „Methods“-Satz)

Wir modellieren Chunk-Grenzen als Change-Points in einem positionsbasierten IKI-Profil (7 IKIs pro 8-Item-Sequenz), d. h. wir suchen eine stückweise konstante Segmentierung über Positionsindex 1…7, wobei Segmente „within-chunk“ und Sprünge „between-chunk/Planungs-Kosten“ abbilden. (Methodischer Rahmen: Kostenfunktion + Suchmethode + Constraint).

## Schritt 0 — Datenaufbereitung (entscheidend für Alters-Heterogenität)

- IKIs berechnen pro korrekter 8-Item-Sequenz: IKI_1..7.
- Robuste QC-Filter (vorher definieren): z. B. IKIs <30 ms oder >2500 ms entfernen; Fehlertrials separat behandeln.
- Log-Transform: x = log(IKI) (IKIs sind meist rechtsschief).
- Motorische Basis korrigieren (sehr stark für Jung/Alt): Pro Person×Tag: Random-Baseline-Profil (Median je Position in Random-Sequenzen). Für strukturierte Sequenzen: y_pos = x_structured_pos − x_rand_pos.

## Schritt 1 — Signaldefinition

Für jede Person×Tag×Bedingung ein 7-dimensionales Profil y = (y_1,…,y_7) (robuste Lage, z. B. Median über alle korrekten Sequenzen).

## Schritt 2 — Kostenfunktion (Truong-Framework)

„Piecewise constant mean“ mit quadratischer Abweichung (L2-Cost). In ruptures: model="l2".

## Schritt 3 — Suchverfahren

PELT (linearer Aufwand). In ruptures: rpt.Pelt(model="l2").

## Schritt 4 — Constraint / Modellselektion

- **Option A:** Penalty über Random-Null kalibrieren (False-Positive-Rate klein).
- **Option B:** Kmax = 3, K über penalized-cost / BIC-artige Regel.

## Schritt 5 — Unsicherheit (Bootstrap)

Boundary-Wahrscheinlichkeit p(pos) pro Position; Chunk-Grenze z. B. p(pos) ≥ 0.6.

## Schritt 6 — Output-Metriken

Pro Person×Tag×Bedingung: # Chunks, Grenzpositionen, Boundary strength, Stabilität p(pos).

## Schritt 7 — Validierung

Konvergenz-Check mit anderen Methoden; Random sollte kaum Grenzen liefern.

---

## Literatur

**Methodik (PELT):**  
Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost. *Journal of the American Statistical Association*, 107(500), 1590–1598. https://doi.org/10.1080/01621459.2012.737745

**CPD-Rahmen:**  
Truong, C., Oudre, L., & Vayatis, N. (2020). Selective review of offline change point detection methods. *Signal Processing*, 167, 107299. https://doi.org/10.1016/j.sigpro.2019.107299

**Software (ruptures):**  
Truong, C., Oudre, L., & Vayatis, N. (2018). ruptures: change point detection in Python. arXiv:1801.00826.

### BibTeX

```bibtex
@article{Killick2012PELT,
  title   = {Optimal Detection of Changepoints with a Linear Computational Cost},
  author  = {Killick, Rebecca and Fearnhead, Paul and Eckley, Idris A.},
  journal = {Journal of the American Statistical Association},
  year    = {2012},
  volume  = {107},
  number  = {500},
  pages   = {1590--1598},
  doi     = {10.1080/01621459.2012.737745}
}

@article{Truong2020Review,
  title   = {Selective review of offline change point detection methods},
  author  = {Truong, Charles and Oudre, Laurent and Vayatis, Nicolas},
  journal = {Signal Processing},
  year    = {2020},
  volume  = {167},
  pages   = {107299},
  doi     = {10.1016/j.sigpro.2019.107299}
}

@article{Truong2018Ruptures,
  title   = {ruptures: change point detection in Python},
  author  = {Truong, Charles and Oudre, Laurent and Vayatis, Nicolas},
  journal = {arXiv preprint arXiv:1801.00826},
  year    = {2018}
}
```

## Links

- PELT (Killick et al., 2012): https://doi.org/10.1080/01621459.2012.737745  
- CPD Review (Truong et al., 2020): https://doi.org/10.1016/j.sigpro.2019.107299  
- ruptures (arXiv): https://arxiv.org/abs/1801.00826  
- ruptures GitHub: https://github.com/deepcharles/ruptures
