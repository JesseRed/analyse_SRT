# Chunking-Methoden: Übersicht

Dieser Ordner enthält ausschließlich **Dokumentation** (kein Code) zu den Chunking-Algorithmen für die SRTT-Analyse.

## Implementierte Methoden

| Methode              | Datei                   | Kurzbeschreibung |
|----------------------|-------------------------|------------------|
| Community Network    | [community_network.md](community_network.md) | Wymbs/Mucha: temporale Multilayer-Community-Detection (Leiden) auf IKI-Ketten. |
| Change-Point (PELT)  | [change_point_pelt.md](change_point_pelt.md) | Change-Point-Detection mit PELT (ruptures), positionsbasiertes IKI-Profil. |
| HSMM Chunking        | [hsmm_chunking.md](hsmm_chunking.md)         | Hidden Semi-Markov Model (hsmmlearn), explizite Zustandsdauern, Fenster-Auswertung. |

## Geplante / Weitere Methoden

- Ranking-Algorithmus (nicht-parametrische Chunk-Kopf-Detektion)
- Bayes’sche HMM
- MDL-basierte Chunking

## Zentrale Voraussetzungen

**[Implementierungsvoraussetzungen.md](Implementierungsvoraussetzungen.md)** – Checkliste und Vertrag für das Implementieren neuer Chunking-Methoden (Eingabe, Schnittstelle, Output, Dokumentation, Registrierung).

## Code

Implementierungen liegen unter `src/chunking/methods/`. Aufruf z. B.:

```bash
python -m src.chunking --method community_network --input-dir SRT --output-dir outputs
python -m src.chunking --benchmark --input-dir SRT --output-dir outputs/benchmark
```

Nach einem Benchmark-Lauf kann die Auswertung mit dem Modul `src.chunking.benchmark_eval` erfolgen (z. B. `load_benchmark_summary`, `comparison_table`, `ari_per_file` für ARI zwischen zwei Methoden pro Datei).
