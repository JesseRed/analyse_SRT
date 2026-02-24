# Community-Network-Methode (Wymbs/Mucha)

## Ziel (Methods-Satz)

Chunk-Grenzen werden als Community-Struktur in einem **temporal multilayer network** modelliert: Pro Trial (Block) wird aus den IKIs einer 8-Item-Sequenz ein gewichteter Ketten-Graph gebaut (Knoten = IKI-Positionen, Kantengewicht = zeitliche Ähnlichkeit benachbarter IKIs). Über alle Trials hinweg werden die Schichten (Layers) mit Kopplung verbunden; mittels Multilayer-Community-Detection (Leiden, Mucha et al.) werden stabile „Communities“ identifiziert, die Chunks entsprechen. Validierung über Permutations-Nullmodell (shuffled IKIs).

**Wichtig:** Ergebnisse werden sowohl aggregiert (`trials.csv`) als auch einzeln pro Teilnehmer unter `artifacts/<filename>/trials.csv` gespeichert, um detaillierte Längsschnitt-Analysen zu ermöglichen.

## Schritte der Pipeline

1. **Daten:** `load_srt_file`, `extract_ikis` (gemeinsame Pipeline). Pro Block: 7 IKIs (Differenzen zwischen 8 Zeitpunkten); Outlier-Filter (z. B. 3 SD pro Position).
2. **Graphen:** Pro Trial: `build_trial_network(ikis)` – gewichtete Kette, Gewicht an Kante (i, i+1) = (d_max - |IKI_i - IKI_{i+1}|) / d_max.
3. **Multilayer-Partition:** `find_partition_temporal` (leidenalg) mit RBConfigurationVertexPartition, Resolution γ, Interlayer-Kopplung C; mehrere Läufe, beste Modularität wählen.
4. **Metriken:** Pro Trial: gewichtete Modularität Q, φ = 1/Q; Chunk-Grenzen = Positionen mit Community-Wechsel; `n_chunks`, `chunk_boundaries`, `community_labels`.
5. **Validierung:** Permutations-Nullmodell (IKI-Reihenfolge pro Block shuffeln), empirisches Q mit Null-Verteilung vergleichen → `p_value_permutation`.

## Output-Metriken (Summary)

- `n_blocks`, `mean_n_chunks`, `mean_q_single_trial`, `mean_phi`, `mean_phi_normalized`
- `empirical_q_multitrial`, `null_q_multitrial_mean`, `p_value_permutation`

## Literatur

- Wymbs, N. F., et al. (2012). Differential recruitment of the sensorimotor putamen and frontoparietal cortex during motor chunking in humans. *Neuron*, 66(4), 571–582.
- Mucha, P. J., Richardson, T., Macon, K., Porter, M. A., & Onnela, J.-P. (2010). Community structure in time-dependent, multiscale, and multiplex networks. *Science*, 328(5980), 876–878. https://doi.org/10.1126/science.1184819
- Software: Leiden (leidenalg), python-igraph.

## Implementierung

`src/chunking/methods/community_network.py` – Einstieg: `run_analysis(filepath, sequence_type, gamma=0.9, coupling=0.03, n_iter=100, n_permutations=100, random_state=None)`.
