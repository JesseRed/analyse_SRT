# Rational Chunking (Wu et al. 2023)

## Ziel
Chunk-Erkennung über einen normativen **Rationality-Ansatz**. Teilnehmer optimieren ein internes Utility-Gleichgewicht zwischen Zeitkosten (Speed), Fehlern (Accuracy) und Repräsentationskomplexität (Chunk-Kosten).

## Theoretischer Hintergrund
Basierend auf der Arbeit von Wu et al. (2023). Das Modell geht davon aus, dass Gehirne Chunking nutzen, um Vorhersagbarkeit zu erhöhen und damit Reaktionszeiten zu senken, solange die statistische Struktur der Sequenz dies ohne übermäßigen Genauigkeitsverlust zulässt.

### Utility-Funktion
Das Modell bewertet jede mögliche Partition $R$ (eine von 128 Bitmasken für eine 8-Item-Sequenz) anhand von:
$$J(R) = \text{NLL}_{timing}(x | R) + \kappa \cdot \text{ErrorCost}(R) + \lambda \cdot C(R)$$

- **Speed (NLL)**: Zeitliche Einsparung durch Chunk-Vorhersage (Gaußsches Timing-Modell).
- **Accuracy ($\kappa$)**: Kosten durch Fehler bei falschen Chunk-Vorhersagen.
- **Complexity ($\lambda$)**: Kosten für das Aufrechterhalten vieler Chunk-Grenzen.
- **Rationality ($\beta$)**: Über eine Softmax-Funktion wird die Wahrscheinlichkeit für jede Partition berechnet.

## Parameter
- `lambda` ($\lambda$): Komplexitätskosten pro Chunk-Grenze.
- `kappa` ($\kappa$): Gewichtung der Fehlerkosten.
- `beta` ($\beta$): Inverse Temperatur (Rationalität).
- `delta` ($\delta$): Erwarteter Zeitvorteil (Speed-up) an Chunk-Grenzen.

## Output
- `expected_n_chunks`: Erwartungswert der Chunk-Anzahl über den Posterior.
- `boundary_probabilities`: Wahrscheinlichkeit einer Chunk-Grenze an jeder der 7 IKI-Positionen.
- `best_partition`: Die Partition mit der höchsten Wahrscheinlichkeit (MAP).

## Implementierung
`src/chunking/methods/rational_chunking.py` -- Einstieg: `run_analysis(filepath, sequence_type, ...)`.
