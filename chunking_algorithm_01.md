Implementierungsplan (Change-Point-Chunking mit PELT + ruptures)
Ziel (klarer „Methods“-Satz)

Wir modellieren Chunk-Grenzen als Change-Points in einem positionsbasierten IKI-Profil (7 IKIs pro 8-Item-Sequenz), d. h. wir suchen eine stückweise konstante Segmentierung über Positionsindex 1…7, wobei Segmente „within-chunk“ und Sprünge „between-chunk/Planungs-Kosten“ abbilden. (Methodischer Rahmen: Kostenfunktion + Suchmethode + Constraint).

Schritt 0 — Datenaufbereitung (entscheidend für Alters-Heterogenität)

IKIs berechnen pro korrekter 8-Item-Sequenz: 
𝐼
𝐾
𝐼
1..7
IKI
1..7
	​

.

Robuste QC-Filter (vorher definieren): z. B. IKIs <30 ms oder >2500 ms entfernen; Fehlertrials separat behandeln. (Grenzen kannst du an deine Task-Specs anpassen; wichtig ist: a priori definieren.)

Log-Transform: 
𝑥
=
log
⁡
(
𝐼
𝐾
𝐼
)
x=log(IKI) (IKIs sind meist rechtsschief).

Motorische Basis korrigieren (sehr stark für Jung/Alt):

Pro Person×Tag: bilde ein Random-Baseline-Profil 
𝑥
~
𝑝
𝑜
𝑠
𝑟
𝑎
𝑛
𝑑
x
~
pos
rand
	​

 (Median je Position in Random-Sequenzen).

Für strukturierte Sequenzen: 
𝑦
𝑝
𝑜
𝑠
=
𝑥
~
𝑝
𝑜
𝑠
𝑠
𝑡
𝑟
𝑢
𝑐
𝑡
𝑢
𝑟
𝑒
𝑑
−
𝑥
~
𝑝
𝑜
𝑠
𝑟
𝑎
𝑛
𝑑
y
pos
	​

=
x
~
pos
structured
	​

−
x
~
pos
rand
	​

.
Interpretation: positionsspezifische Zusatz-Kosten gegenüber motorischer Basis → besserer Chunk-Kontrast als rohe IKIs.

Warum das wissenschaftlich gut ist: Du entkoppelst „generell langsamer“ von „positionsspezifischer Planungs-Peak“. Das reduziert genau den Hauptkritikpunkt bei IKI-Schwellenmethoden in heterogenen Gruppen.

Schritt 1 — Signaldefinition (was segmentierst du genau?)

Für jede Person×Tag×Bedingung (z. B. häufige vs seltene Sequenz) erzeugst du ein 7-dimensionales Profil:

𝑦
=
(
𝑦
1
,
…
,
𝑦
7
)
y=(y
1
	​

,…,y
7
	​

) (robuste Lage, z. B. Median über alle korrekten Sequenzen dieses Tages / dieser Bedingung).
Optional (wenn du willst): Profile blockweise (z. B. je 10–20 Sequenzen) → dann kannst du Chunk-Grenzen „über Training“ als Zeitreihe verfolgen.

Schritt 2 — Kostenfunktion (Truong-Framework: “cost function”)

Wähle „piecewise constant mean“ mit quadratischer Abweichung (L2-Cost):

Annahme: innerhalb eines Chunks sind 
𝑦
𝑝
𝑜
𝑠
y
pos
	​

 ähnlich; zwischen Chunks gibt es einen Sprung in Mittelwert.
Das ist Standard in offline CPD und direkt kompatibel mit PELT.

Praktisch in ruptures: model="l2".

Schritt 3 — Suchverfahren (Truong-Framework: “search method”)

Nutze PELT (linearer Aufwand, penalisiertes Optimierungsproblem).
Praktisch in ruptures: rpt.Pelt(model="l2").

Warum PELT hier gut passt:

du hast viele Personen×Tage×Bedingungen → insgesamt viele CPD-Fits (PELT ist effizient),

du willst eine saubere, zitierbare Methodik statt Heuristik.

Schritt 4 — Constraint / Modellselektion (der Punkt, der Reviewer überzeugt)

Du brauchst eine a-priori Regel, wie viele Change-Points erlaubt sind. Zwei solide Optionen:

Option A (mein Favorit): Penalty kalibrieren über Random-Null

Fitte CPD auf Random-Profilen 
𝑦
𝑟
𝑎
𝑛
𝑑
y
rand
 (wo keine stabile Chunk-Struktur erwartet ist).

Wähle die Penalty so, dass die False-Positive-Rate klein ist (z. B. im Random im Mittel ≤0.1 Change-Points pro Profil oder ≤5% Profile mit irgendeinem Change-Point).

Fixiere diese Penalty anschließend und wende sie auf strukturierte Profile an.

Vorteil: Penalty ist daten-geleitet, aber über Null-Bedingung (nicht über die zu testende Struktur), und damit gut begründbar.

Option B: Fixe Obergrenze + IC-Regel

Setze Kmax = 3 (bei 7 Punkten ist mehr sowieso kaum sinnvoll) und wähle 
𝐾
K über eine einfache penalized-cost / BIC-artige Regel.
Hier kannst du dich explizit auf „penalized cost minimization“ in CPD beziehen.

Schritt 5 — Unsicherheit / Stabilität (Bootstrap statt „hartes“ Ergebnis)

Weil dein Profil nur 7 Werte hat, ist Stabilität zentral:

Bootstrap innerhalb Person×Tag×Bedingung: resample Sequenzen, bilde Profil, fitte CPD.

Ergebnis: Boundary-Wahrscheinlichkeit pro Position 
𝑝
(
𝑝
𝑜
𝑠
)
p(pos).

Definiere Chunk-Grenze als Positionen mit z. B. 
𝑝
(
𝑝
𝑜
𝑠
)
≥
0.6
p(pos)≥0.6 (a priori).

Das ist extrem reviewer-robust: du berichtest nicht nur „Grenze bei pos=3“, sondern „pos=3 mit 0.78 Stabilität“.

Schritt 6 — Output-Metriken (direkt publikationsfähig)

Pro Person×Tag×Bedingung:

# Chunks (= #ChangePoints + 1)

Grenzpositionen (z. B. [3,5])

Boundary strength (Sprunghöhe in 
𝑦
y über die Grenze, z. B. ΔMean)

Stabilität 
𝑝
(
𝑝
𝑜
𝑠
)
p(pos)

optional: Reorganisation über Tage (Edit-Distance zwischen Partitionen Tag1→Tag2→Tag3)

Dann kannst du Alterseffekte / Trainingseffekte auf diese Metriken modellieren (LMM/GEE etc.).

Schritt 7 — Validierung (kurzer „Sanity-Check“ Abschnitt)

Konvergenz-Check: liegen CPD-Grenzen dort, wo IED/Nonparam-Rang häufig Peaks sieht? (nicht identisch, aber konsistent)

Manipulations-Check: Random sollte kaum Grenzen liefern (falls doch: Penalty hoch oder QC/Normalisierung anpassen).

Kann man „einfach ruptures nehmen“?

Ja — als Software. Wissenschaftlich sauber wird es, wenn du klar schreibst:

Algorithmus: PELT (Killick et al.)

CPD-Rahmen (Kostenfunktion/Suche/Constraint): Truong et al. Review

Implementierung: ruptures (Truong et al., arXiv)

Vollständige Zitation (manuskriptfertig)

Methodik (PELT):
Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost. Journal of the American Statistical Association, 107(500), 1590–1598. https://doi.org/10.1080/01621459.2012.737745

CPD-Rahmen/Überblick (zur Einordnung deiner Design-Choices):
Truong, C., Oudre, L., & Vayatis, N. (2020). Selective review of offline change point detection methods. Signal Processing, 167, 107299. https://doi.org/10.1016/j.sigpro.2019.107299

Software (ruptures):
Truong, C., Oudre, L., & Vayatis, N. (2018). ruptures: change point detection in Python. arXiv:1801.00826.

Optional in Methods (Software-Angabe): “Implemented in Python using the ruptures package (version x.y.z).”


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
Links (falls du sie 1:1 brauchst)
PELT (Killick et al., 2012): https://doi.org/10.1080/01621459.2012.737745
CPD Review (Truong et al., 2020): https://doi.org/10.1016/j.sigpro.2019.107299
ruptures paper (arXiv): https://arxiv.org/abs/1801.00826
ruptures GitHub: https://github.com/deepcharles/ruptures