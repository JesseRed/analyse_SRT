# Basis-Bewertung der 5 Chunking-Methoden auf dem Kernset

**Stand:** 2026-10-11 · **Daten:** `simulations/core` (592 Dateien) · **Methoden-Outputs:** `outputs/sim_core` (nicht in Git)
**Reproduzieren:**
```bash
python -m src.simulation run-methods --sim-dir simulations/core --results outputs/sim_core --jobs 14
python -m src.simulation run-methods --sim-dir simulations/core --results outputs/sim_core --methods hsmm_chunking --sequences blue --jobs 14
python -m src.simulation evaluate    --sim-dir simulations/core --results outputs/sim_core
```
Die Tabellen hier stammen aus `outputs/sim_core/evaluation/` (`evaluation_summary.csv`, `static_power.csv` usw.). Eine Kopie der Zusammenfassung liegt in [`baseline_evaluation_summary.csv`](baseline_evaluation_summary.csv).

**Einstellungen**
- Alle Methoden laufen mit `methods_config.json` und `n_null_runs = 0`.
- HSMM lief nur auf Blue (etwa 45 s pro Datei).
- Community Network: Der veraltete Config-Schlüssel `n_permutations` wurde ignoriert.

> **Vorläufig.** Das Kernset hat pro Zelle nur wenige Probanden (Familie B: 2 Probanden × 2 Sitzungen). Einzelwerte schwanken deshalb stark, Monte-Carlo-SE siehe `*_mcse` in der CSV.
> - **Belastbar** sind nur die großen, durchgängigen Muster.
> - **Inferenzielle Aussagen** erfordern das volle Set (`configs/simulation/full.yaml`).

Alle Maße gelten für **Blue**, sofern nicht anders angegeben. „Grenze erkannt“ wird konventions-bewusst bewertet (`*_conv`). Segment-Methoden bekommen einen Treffer, wenn sie *k*−1 oder *k* melden.

---

## 1. Fehlalarme ohne Chunks (Familie A)

Anteil der Blöcke ohne wahre Grenze, in denen die Methode trotzdem mindestens eine Grenze meldet:

| Zelle | PELT | Community Network | HCRP-LM | HSMM | Rational |
|---|---|---|---|---|---|
| A1 reines Rauschen | 0,00 | **1,00** | **1,00** | 0,00 | 0,00 |
| A2 + Motorik (Position, Tastenpaar) | 0,01 | **1,00** | **1,00** | 0,00 | 0,03 |
| A3 + Lern-/Ermüdungstrend | 0,01 | **1,00** | **1,00** | 0,00 | 0,02 |
| A4 + Heterogenität, Lapses | 0,01 | **1,00** | **1,00** | 0,00 | 0,02 |
| A5 + Sequenz-Speedup (realistisch) | 0,00 | **1,00** | **1,00** | 0,00 | 0,05 |
| *Green, Mittel A1–A5* | 0,01 | 1,00 | 1,00 | – | **0,23** |
| *Yellow, alle Familien* | 0,00 | 1,00 | 0,99 | – | 0,10 |

**Community Network und HCRP-LM melden immer Grenzen**, auch in reinem Rauschen.
- Bei CN liegt das an der Normierung pro Block. Die Leiden-Partition zerlegt jede Kette in etwa 2 Communities.
- Bei HCRP liegt es an der z-Schwelle innerhalb des Blocks über einem Surprisal-Profil, das nie flach ist.

**Rational ist in Blue gut kalibriert.** In Green meldet es aber in **23 % (bis 34 %)** der chunkfreien Blöcke eine Grenze, sobald realistische Tastenpaar-Effekte im Spiel sind (A2–A5; A1 ohne Motorik: 3 %). Die motorisch langsame Position 4 von Green wird als Chunk-Grenze gelesen.

## 2. Erkennbarkeit nach Effektstärke (Familie B, Spike-Kanal)

**AUC:** Trennbarkeit von Dateien mit und ohne Chunks anhand der mittleren Anzahl gemeldeter Grenzen. 0,5 = Zufall.

| δ (log) | PELT | CN | HCRP | HSMM | Rational |
|---|---|---|---|---|---|
| 0,05 | 0,62 | 0,56 | 0,52 | 0,56 | 0,31 |
| 0,10 | 0,46 | 0,40 | 0,19 | 0,48 | 0,53 |
| 0,20 | 0,50 | 0,57 | 0,31 | 0,44 | 0,67 |
| 0,30 | 0,50 | 0,58 | 0,33 | 0,44 | **0,88** |
| 0,50 | 0,61 | 0,33 | 0,28 | 0,52 | **1,00** |

**Recall** (Anteil gefundener wahrer Grenzen), gemittelt über die Rauschniveaus:

| δ (log) | PELT | CN | HCRP | HSMM | Rational |
|---|---|---|---|---|---|
| 0 | 0,00 | 0,59 | 0,54 | 0,00 | 0,01 |
| 0,20 | 0,00 | 0,57 | 0,31 | 0,00 | 0,16 |
| 0,30 | 0,00 | 0,60 | 0,71 | 0,00 | 0,25 |
| 0,50 | 0,02 | 0,58 | 0,22 | 0,00 | **0,61** |

- **Rational** ist die einzige Methode, deren Leistung mit der Effektstärke steigt. d50 (50 % Recall) liegt bei δ ≈ **0,37–0,39 log**, die Grenz-IKI muss also etwa 45 % langsamer sein. Bei hohem Rauschen wird 50 % nicht erreicht.
- **PELT** findet selbst bei δ = 0,5 praktisch nichts, obwohl das Signal im Profil Blue − Yellow klar sichtbar ist. Ursache ist die zu konservative Penalty-Kalibrierung.
- **HSMM** meldet in **keinem** Block eine Grenze, weder in A noch in B. Das passt zum Duration-Prior mit Mittelwert 7 = Sequenzlänge.
- **Community Network** liegt bei einer Trefferquote von etwa 0,6, unabhängig vom Effekt. Das ist Zufallsniveau, weil die Methode immer etwa 1–2 Grenzen setzt.
- **HCRP-LM** hängt nicht von den Zeiten ab. Seine Treffer kommen nur dadurch zustande, dass die gezogene Struktur zufällig zu den Surprisal-Spitzen der Stimulusfolge passt. Das ist die erwartete **Negativkontrolle**.

**Signaltyp** (AUC bei δ = 0,5):

| Kanal | PELT | CN | HCRP | HSMM | Rational |
|---|---|---|---|---|---|
| Spike an der Grenze | 0,61 | 0,33 | 0,28 | 0,52 | **1,00** |
| Niveau-Unterschied zwischen Chunks | 0,73 | 0,35 | 0,34 | 0,54 | **0,98** |
| geteiltes Rauschen im Chunk (Korrelation) | 0,71 | 0,74 | 0,54 | 0,61 | 0,35 |

Beim Niveau-Kanal erkennt Rational zwar, *dass* Struktur da ist (AUC 0,98). Es setzt die Grenzen aber an die falschen Stellen (Recall ≈ 0), weil sein Modell Spikes annimmt. Den Korrelations-Kanal (Acuna et al. 2014) erkennt **keine** Methode zuverlässig.

## 3. Dynamik, Gruppen, Robustheit

- **Verlauf über Sitzungen (E_realistic):** Die Spearman-Korrelation zwischen wahrer und geschätzter Chunk-Zahl liegt bei Rational bei 0,65, sonst zwischen −0,55 (CN) und 0,25. Familie C hat im Kernset nur 3 Probanden pro Zelle und ist noch nicht interpretierbar.
- **Gruppen (D, jung vs. alt; in Wahrheit hat „alt“ 1,2–1,3 Chunks mehr pro Block):** Keine Methode findet den Unterschied. Community Network zeigt bei δ = 0,15 sogar einen **signifikanten Effekt in die falsche Richtung** (p = 0,04; „alt“ hat weniger Chunks). Ein Gruppenbefund mit CN kann also durch Tempo- und Rauschunterschiede entstehen, ohne dass sich das Chunking unterscheidet. Bei nur 1 Replikat × 6 Probanden ist das ein Warnsignal, keine Schätzung der Teststärke.
- **Datenqualität (F):** Die F1-Werte liegen bei CN 0,47–0,62, Rational 0,27–0,34, HCRP 0,16–0,38, PELT und HSMM ≈ 0. Der hohe Wert von CN ist **kein Gütemerkmal**: Er entsteht, weil die Methode immer Grenzen setzt und bei 2–4 wahren Grenzen pro Block zufällig trifft (vgl. Fehlalarmrate 1,0 und Recall unabhängig von δ). Ein F1 allein ist deshalb nie ausreichend; es muss immer zusammen mit Familie A gelesen werden.

## 4. Konsequenzen

1. **Keine der fünf Methoden ist in der aktuellen Form ein valides Chunking-Maß für diese Daten.**
   - Rational reagiert als einzige auf echte Chunk-Signale. Es ist aber in Green motorisch konfundiert, erkennt nur Spikes und braucht große Effekte.
2. Die Befunde aus den echten Daten lassen sich jetzt erklären:
   - PELT, HSMM und Rational finden „etwa 1 Chunk“.
   - CN findet „2–2,4 Chunks, egal welche Sequenz“.
   - Beides sind **Eigenschaften der Methoden**, keine Aussagen über die Probanden.
3. **Priorisierte Korrekturen** (siehe README, „Bekannte Einschränkungen“), danach Neubewertung auf `full`:
   - PELT: Penalty-Kalibrierung an Nullprofilen gleicher Blockzahl.
   - HSMM: Duration-Prior.
   - Rational: Motorik-Baseline aus Tastenpaar-Effekten statt nur Fenstermittelwert.
   - CN: Nullmodell mit gleicher `n_iter` und eine Entscheidungsregel, die auch „keine Grenze“ erlaubt.
   - HCRP: Parameter an RTs anpassen (wie bei Éltető et al. 2022).
4. Neue Methoden sollten mindestens folgende Kriterien erfüllen:
   - **Fehlalarmrate ≤ 5 % in A2–A5**, ausdrücklich auch bei Green.
   - Steigender Recall über δ.
   - Den wahren Gruppenunterschied in D erkennen.
