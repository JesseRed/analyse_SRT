# Change-Point-Chunking (PELT + ruptures)

Implementierungsplan und Referenz zur Implementierung in `src/chunking/methods/change_point_pelt.py`.

---

## Dynamisches Chunking mit Sliding Windows (PELT-CPD über Positionsachse)

### 1) Grundidee: drei getrennte Profile pro Zeitfenster (Interleaving egal)
Wir definieren globale Zeitfenster über die Block-Reihenfolge (`BlockNumber`) und berechnen innerhalb jedes Fensters getrennte Positionsprofile für drei Bedingungen:

- **blue**: strukturierte Sequenz A  
- **green**: strukturierte Sequenz B  
- **yellow**: random (Nullbedingung)

Da die Bedingungen blockweise gemischt sind, repräsentiert jedes Fenster einen vergleichbaren „Tageszustand“ (Warm-up, Fatigue, Aufmerksamkeit) und eignet sich für dynamische Vergleiche.

**Default-Parameter** (für 120 Blöcke/Tag gut passend):  
- Sliding window: \(W = 30\) Blöcke, Schrittweite \(S = 10\)  
- Mindestanzahl pro Fenster×Bedingung: `min_blocks = 6` (nur korrekte Sequenzen)

---

### 2) Aus Rohzeilen → pro Block ein IKI-Vektor (Länge 7)
Pro `BlockNumber`:

1. Nach `EventNumber` sortieren.  
2. Nur Blöcke behalten, die **alle 8 Hits** enthalten (`isHit == 1` für alle 8 Events).  
3. IKI pro Position berechnen (aus *Time since block start*):
\[
IKI_{\text{pos}} = t_{\text{pos}+1} - t_{\text{pos}}, \quad \text{pos}=1..7
\]
4. Log-Transformation:
\[
x_{\text{pos}} = \log(IKI_{\text{pos}})
\]

**Ergebnis pro Block:** `logIKI_1..logIKI_7`, plus `sequence` (blue/green/yellow), plus `BlockNumber`.

---

### 3) Pro Fenster getrennte Positionsprofile bauen
Für jedes Fenster \(b\) (30 Blöcke) und jede Bedingung \(c \in \{\text{blue, green, yellow}\}\):
\[
\tilde{x}_{b,\text{pos}}^{(c)} = \text{median}\left(\log(IKI_{\text{pos}})\right)
\]
Das liefert pro Fenster drei 7-Vektoren:
\(\tilde{x}_b^{blue}, \tilde{x}_b^{green}, \tilde{x}_b^{yellow}\).

#### Random-Baseline-Korrektur (empfohlen)
Da **yellow** die Nullbedingung ist, definieren wir für strukturiert:
\[
y_{b,\text{pos}}^{blue} = \tilde{x}_{b,\text{pos}}^{blue} - \tilde{x}_{b,\text{pos}}^{yellow}
\]
\[
y_{b,\text{pos}}^{green} = \tilde{x}_{b,\text{pos}}^{green} - \tilde{x}_{b,\text{pos}}^{yellow}
\]

**Interpretation:** positionsspezifische Zusatzkosten über random hinaus (trennt „global langsamer“ von „Chunk-Peak“).

**Yellow** bleibt als Nullsignal und wird direkt auf \(\tilde{x}_b^{yellow}\) analysiert (sollte wenig/instabil chunked sein).

---

### 4) Change-Point-Chunking pro Fenster und pro Bedingung (PELT)
CPD läuft über die **Positionsachse** \(1..7\), nicht über Zeit.

- **yellow:** CPD auf \(\tilde{x}_b^{yellow}\)  
- **blue/green:** CPD auf \(y_b^{blue}\) bzw. \(y_b^{green}\)

**Settings:**
- Cost: L2 (piecewise-constant mean, Standard) oder RBF (kernel-basiert, sensitiver für Verteilungsänderungen; Truong C12)
- Search: PELT
- Output: Change-point Positionen, \(\#\text{chunks} = \#CP + 1\)

**Cost-Funktionen (Truong 2020):**
- **L2** (Default): piecewise constant mean, etabliert für IKI-Daten
- **RBF**: kernel-basiert, sensitiver für subtile Änderungen; optional `--rbf-gamma` für Bandbreite (sonst Median-Heuristik)

---

### 5) Penalty-Kalibrierung über Nullbedingungen (sauber begründet)
Ziel: Penalty \(\lambda\) so wählen, dass in der Nullbedingung kaum falsche Grenzen entstehen.

#### 5A) Penalty für yellow (Nullsignal)
Kalibriere \(\lambda\) so, dass CPD auf \(\tilde{x}_b^{yellow}\) eine kleine False-Positive-Rate hat, z. B.:

- ≤ 5% der Fenster haben ≥ 1 Change-point  
  **oder**
- ≤ 0.1 Change-points pro Fenster im Mittel

#### 5B) Penalty für baseline-korrigierte Signale (blue/green)
Wichtig: \(y = \tilde{x}^{structured} - \tilde{x}^{yellow}\) hat eine andere Varianz als yellow allein. Dafür nutzen wir einen robusten „Null-Trick“ innerhalb jedes Fensters:

1. Splitte die **yellow**-Blöcke im Fenster zufällig in zwei Hälften \(A,B\).  
2. Berechne:
\(\tilde{x}_{b}^{yellow,A}\) und \(\tilde{x}_{b}^{yellow,B}\)  
3. Bilde das „random-minus-random“-Nullsignal:
\[
y_b^{null} = \tilde{x}_b^{yellow,A} - \tilde{x}_b^{yellow,B}
\]

Dieses \(y_b^{null}\) hat:
- gleichen Fensterzustand,
- ähnliche Schätzvarianz wie baseline-korrigierte Profile,
- **keine echte Chunk-Struktur** (random vs random).

Kalibriere \(\lambda\) so, dass CPD auf \(y_b^{null}\) wieder die gewünschte FPR erfüllt.

**Kurz:**  
- `penalty_yellow` aus \(\tilde{x}_b^{yellow}\)  
- `penalty_structured` aus \(y_b^{null}\)

---

### 6) Bootstrap-Stabilität pro Fenster×Bedingung
Um die Dynamik überzeugend zu quantifizieren:

- Pro Fenster und Bedingung bootstrappst du Blöcke (z. B. 500×),
- baust Profil,
- führst CPD aus,
- erhältst für jede Position:
\[
p_{b,c}(\text{pos}) = \Pr(\text{Change-point bei pos} \mid \text{Fenster } b, \text{Bedingung } c)
\]

Damit kannst du für blue/green/yellow **Heatmaps** (Fenster × Position) darstellen.

---

### 7) Dynamik-Metriken (publikationsnah)
Pro Bedingung \(c\):

- **Boundary-Heatmap:** \(p_{b,c}(\text{pos})\)
- **Drift** (Fenster-zu-Fenster Änderung der Boundary-Verteilung):
\[
\sum_{\text{pos}} \left| p_{b+1,c}(\text{pos}) - p_{b,c}(\text{pos}) \right|
\]
- **Stabilisierung:** Entropie von \(p_{b,c}\) nimmt ab (Boundary-Masse konzentriert sich)
- **Chunk-count trajectory:** \(\#\text{chunks}\) pro Fenster (MAP oder Erwartungswert aus Bootstrap)
- Optional **Reorganization events** (Merge/Split/Shift) über Vergleich der MAP-Partitionen zwischen Fenstern

---

### 8) Praktische Output-Hinweise und Presets

**CLI-Presets:**
- `--penalty-sensitive`: weniger konservative Penalty (fpr_target=0.10, mean_cp_target=0.2) für mehr Change-Points
- `--short-session`: kleinere Fenster (window_size=20, step=5, min_blocks=4) für kurze Sessions, damit mehr gültige Fenster entstehen

**Empfehlung:** Primäranalyse mit Defaults; Sensitivitätsanalysen mit Presets bei Probanden mit wenig Chunk-Struktur oder wenigen Fenstern.

- `validation`-Artefakte (z. B. Boundary-Heatmap, Drift, window_n_chunks) können als `validation.json` gespeichert werden:
  - Single-File: direkt im Output-Verzeichnis.
  - Batch: pro Datei unter `artifacts/<source_stem>/validation.json`.
- Empfohlenes Qualitätsflag im Summary: `reliable = (n_windows_valid >= 3)` für robuste Filterung in Folgeanalysen.
- Optionaler Fallback für kurze Sessions: Falls keine ausreichenden Yellow-Fenster/Nullsignale für die Penalty-Kalibrierung verfügbar sind, kann ein fixer Fallback-Penalty verwendet werden (mit Kennzeichnung im Output, z. B. `penalty_fallback_used`).

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



### References

- Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost. *Journal of the American Statistical Association, 107*(500), 1590–1598. https://doi.org/10.1080/01621459.2012.737745  
- Truong, C., Oudre, L., & Vayatis, N. (2020). Selective review of offline change point detection methods. *Signal Processing, 167*, 107299. https://doi.org/10.1016/j.sigpro.2019.107299  
- Truong, C., Oudre, L., & Vayatis, N. (2018). *ruptures: change point detection in Python*. arXiv:1801.00826. https://arxiv.org/abs/1801.00826
PELT paper (JASA): https://doi.org/10.1080/01621459.2012.737745
CPD review (Signal Processing): https://doi.org/10.1016/j.sigpro.2019.107299
ruptures paper (arXiv): https://arxiv.org/abs/1801.00826
ruptures PELT docs (mentions Killick2012): https://centre-borelli.github.io/ruptures-docs/user-guide/detection/pelt/
ruptures GitHub: https://github.com/deepcharles/ruptures