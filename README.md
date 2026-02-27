# analyse_SRT – Chunking-Analysen für die Serial Reaction Time Task (SRTT)

Dieses Projekt wertet SRTT-Daten (8-Item-Sequenzen, Inter-Key-Intervals, IKIs) mit verschiedenen **Chunking-Methoden** aus. Mehrere Algorithmen können auf denselben Daten laufen und ihre Ergebnisse werden einheitlich ausgegeben, sodass sie wissenschaftlich verglichen und weiterverarbeitet werden können.

---

## Voraussetzungen und Setup

- **Python 3.11** (oder kompatibel)
- Virtuelle Umgebung empfohlen (steht in `.gitignore`)

### Erste Einrichtung (nach Clone)

```bash
cd analyse_SRT
python3 -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

Ohne Aktivierung der venv:

```bash
.venv/bin/python -m src.chunking ...
```

---

## Projektstruktur

```
analyse_SRT/
├── README.md                 # Diese Datei
├── requirements.txt
├── methods_config.json       # Standard-Parameter für alle Algorithmen
├── infos.md                  # Kontext: SRTT, 60/30/30-Design, Literatur
│
├── algorithms/               # Dokumentation der Methoden
│   ├── community_network.md
│   ├── change_point_pelt.md
│   ├── hsmm_chunking.md
│   ├── hcrp_lm.md
│   └── rational_chunking.md   # Neu: Wu et al. (2023)
│
├── src/
│   └── chunking/             # Core-Logik
│       ├── _base.py           # ChunkingResult, Schema
│       ├── run.py             # Single/Batch/Benchmark-Runner
│       └── methods/           # Implementierung der 5 Methoden
│           ├── community_network.py
│           ├── change_point_pelt.py
│           ├── hsmm_chunking.py
│           ├── hcrp_lm.py
│           └── rational_chunking.py
│
├── SRT/                      # Eingabedaten
└── outputs/                  # Analyseergebnisse (automatisch erstellt)
```

- **Eingabedaten:** CSV-Dateien im Ordner `SRT/` (oder anderer Pfad) mit Spalten u. a. `BlockNumber`, `EventNumber`, `Time Since Block start`, `isHit`, `sequence`. Details siehe `algorithms/Implementierungsvoraussetzungen.md`.

---

## Analysen starten

Alle Aufrufe erfolgen über das Chunking-Modul. Die Ausgabe landet jeweils in einem **eigenen Unterordner** unter dem angegebenen `--output-dir`.

### 1. Eine Methode, eine einzelne Datei

```bash
python -m src.chunking \
  --method community_network \
  --input-file SRT/Beispiel_Teilnehmer.csv \
  --output-dir outputs/mein_lauf
```

Ergebnisse: `outputs/mein_lauf/community_network/` (siehe Abschnitt „Ausgabe“).

### 2. Eine Methode, alle Dateien (Batch)

```bash
python -m src.chunking \
  --method community_network \
  --input-dir SRT \
  --output-dir outputs/batch_blau \
  --pattern "*.csv" \
  --sequence-type blue \
  --limit 10
```

- `--input-dir`: Ordner mit den CSV-Dateien  
- `--pattern`: Glob für Dateinamen (Standard: `*.csv`)  
- `--sequence-type`: `blue`, `green`, `yellow` oder `all` (alle drei getrennt)  
- `--limit`: optional, maximale Anzahl Dateien (z. B. zum Testen)

Methodenspezifische Optionen (Beispiele):

- **Community Network:** `--gamma 0.9`, `--coupling 0.03`, `--n-iter 20`
- **Change Point (PELT):** `--penalty-sensitive`, `--cost-model l2|rbf`
- **HCRP-LM:** `--n-levels 3`, `--strength 0.5`, `--threshold-z 1.0`
- **Rational Chunking:** `--lam 1.0`, `--kappa 0.5`, `--beta 5.0`

Für Details zu den Parametern siehe die jeweilige Dokumentation unter `algorithms/`.

### 3. Alle Methoden gleichzeitig (Benchmark / Run-All)

Um alle verfügbaren Methoden auf einmal auszuführen (Benchmark-Modus), gibt es zwei Möglichkeiten:

**Variante A: Einfacher Aufruf aller Methoden**
```bash
python -m src.chunking --method all --input-dir SRT --limit 5
```

**Variante B: Mit Konfigurationsdatei (Empfohlen)**
Erstelle eine `methods_config.json` mit den gewünschten Parametern für jede Methode und übergebe sie mit `--config`:
```bash
python -m src.chunking --method all --config methods_config.json --input-dir SRT
```

- **Zusammenführung:** Die Datei `benchmark_summary.csv` im Output-Verzeichnis enthält eine konsolidierte Tabelle aller Ergebnisse (eine Zeile pro Datei × Methode).
- **Validierung im Benchmark:** Du kannst `--n-null-runs` auch im Benchmark-Modus verwenden, um alle Methoden gegen Null-Modelle zu prüfen.

### 4. Statistische Validierung (Null-Modelle)

Alle Methoden unterstützen nun einen Vergleich gegen ein **Null-Modell** (zufällig permutierte IKIs). Dabei werden $p$-Werte und $z$-Scores berechnet, um zu zeigen, dass die gefundene Struktur (z. B. Modularität $Q$) signifikant über dem Zufallsniveau liegt.

```bash
python -m src.chunking \
  --method community_network \
  --input-file SRT/Beispiel.csv \
  --n-null-runs 100 \
  --output-dir outputs/validation_run
```

- `--n-null-runs`: Anzahl der Randomisierungen (Standard: 100). Setze auf 0, um die Validierung zu deaktivieren.
- **Ergebnisse:** Die Metriken (`p_value`, `z_score`, `null_mean`, etc.) erscheinen in der `summary.csv` und detailliert in `validation.json` im Artefakt-Ordner des Probanden.

### 5. Verfügbare Methoden anzeigen

Aktuell sind folgende Methoden implementiert:
1. **community_network**: Netzwerkbasierte Partitionierung (Wymbs/Mucha).
2. **change_point_pelt**: Change-Point-Detektion auf IKI-Zeitreihen.
3. **hcrp_lm**: Hierarchical Chinese Restaurant Process (Action Chunking).
4. **hsmm_chunking**: Hidden Semi-Markov Modelle zur Chunk-Identifikation.
5. **rational_chunking**: Normatives Nutzen-Modell (Wu et al. 2023).

Die Liste der Registrierten Methoden kann via `python -m src.chunking --help` eingesehen werden.

---

## Ausgabe: Was wird wo geschrieben?

Jeder Lauf (Single-File, Batch oder eine Methode im Benchmark) schreibt in **einen** Ordner mit festem Schema:

```
outputs/<dein_output_dir>/<method_name>/
├── meta.json          # Lauf-Infos: Methode, Zeitstempel, Git-Commit, n_files, …
├── parameters.json    # Alle Laufparameter (inkl. methodenspezifisch)
├── summary.csv        # Eine Zeile pro Quelldatei (siehe unten)
├── trials.csv         # Eine Zeile pro Trial/Block
├── errors.csv         # Fehlgeschlagene Dateien (source_file, error)
└── progress.log       # Fortschrittslog (bei Batch/Benchmark)
```

### summary.csv

- **Mindestspalten:** `source_file`, `sequence_type`, `method`, `n_blocks`  
- **Kern-Metriken:** `mean_n_chunks` (Durchschnittliche Anzahl Chunks pro Block).  
- **Metadaten:** `participant_id`, `session`, `day_index` (automatisch extrahiert).
- **Methodenspezifisch:** z. B. `avg_boundary_probs` (Vektor der Wahrscheinlichkeiten), `mean_q` (Modulartität), etc.

Eine Zeile = eine analysierte Teilnehmerdatei. Ideal für übergreifende Auswertungen (z. B. nach Gruppe, Tag, Bedingung).

### trials.csv

- **Mindestspalten:** `source_file`, `sequence_type`, `method`, `block_number`
- **Chunking-Ergebnisse:** 
  - `n_chunks`: Identifizierte Anzahl Chunks (oder Erwartungswert).
  - `chunk_boundaries`: Liste der Grenz-Indizes (1–7).
  - `boundary_probs`: Punktweise Wahrscheinlichkeiten f. Grenzen (falls unterstützt).
- **Metadaten:** `participant_id`, `session`, `day_index`.
- **Methodenspezifisch:** z. B. `phi`, `community_labels`, `ikis`, `log_likelihood`.

Eine Zeile = ein Block/Trial. Für trialweise Auswertungen und Methodenvergleiche (z. B. ARI).

### meta.json / parameters.json

- **Reproduzierbarkeit:** Zeitstempel, optional Git-Commit, Eingabe-Pfad/Pattern, alle Parameter.  
- Für Methodenpapiere und Supplement: Welche Softwareversion, welche Einstellungen.

### validation.json / artifacts (optional)

- Falls eine Methode `result.validation` liefert, wird sie als JSON persistiert.
- **Detaillierte Artefakte:** Methoden wie `hcrp_lm` und `rational_chunking` speichern pro Quelldatei detaillierte Analysen unter `artifacts/<source_stem>/`:
  - `trials.csv`: Trial-weise Ergebnisse.
  - `validation.json`: Diagnostische Informationen (z. B. Modell-Fits).

### combined_summary.csv (optional bei `--sequence-type all`)

- Über `--merge-summaries` kann aus `blue/summary.csv`, `green/summary.csv`, `yellow/summary.csv` eine `combined_summary.csv` erzeugt werden (inkl. Delta-Spalten gegen `yellow`).

### Auswerteroutine (Chunking über Tage und Sequenzen)

Nach einem Lauf mit `--sequence-type all` (z. B. `change_point_pelt`) kannst du einen wissenschaftlich aufbereiteten Bericht erzeugen: Verlauf des Chunkings über die Tage (pro Proband und Sequenz) sowie Unterschiede zwischen den Sequenzen (Blue/Green vs. Yellow).

```bash
python -m src.chunking.output_evaluation outputs/SRT_Test2/change_point_pelt --csv outputs/SRT_Test2/change_point_pelt/evaluation_data.csv
```

Es werden geschrieben: `evaluation_report.md` (im Methoden-Output-Verzeichnis) und optional eine CSV mit der zusammengeführten Tabelle. Programmgesteuert:

```python
from src.chunking.output_evaluation import evaluate_outputs, load_merged_summary, build_evaluation_report

result = evaluate_outputs(
    "outputs/SRT_Test2/change_point_pelt",
    output_csv_path="outputs/SRT_Test2/change_point_pelt/evaluation_data.csv",
)
# result["report_path"], result["csv_path"], result["participants"]
```

---

## Benchmark-Auswertung (Vergleich mehrerer Methoden)

Nach einem `--benchmark`-Lauf kannst du das Modul `benchmark_eval` nutzen:

```python
from src.chunking import benchmark_eval

# Gemeinsame Summary-Tabelle (eine Zeile pro Datei × Methode)
df = benchmark_eval.load_benchmark_summary("outputs/benchmark")

# Ausgewählte Spalten für Vergleiche
tab = benchmark_eval.comparison_table("outputs/benchmark")

# ARI zwischen zwei Methoden pro Datei (wenn beide chunk_boundaries liefern)
ari_df = benchmark_eval.ari_per_file(
    "outputs/benchmark",
    method_a="community_network",
    method_b="rational_chunking",
)
```

---

## Neue Chunking-Methoden hinzufügen

1. **Implementierungsvorschlag** als MD-Datei anlegen (z. B. unter `algorithms/meine_methode.md`: Ziel, Schritte, Literatur).
2. **Implementierungsvoraussetzungen** durchgehen:  
   `algorithms/Implementierungsvoraussetzungen.md`
3. **Implementierung** in `src/chunking/methods/<method_name>.py` mit Funktion  
   `run_analysis(filepath, sequence_type, **kwargs) -> ChunkingResult`
4. **Registrierung** in `src/chunking/methods/__init__.py`:  
   `REGISTRY["method_name"] = run_analysis`
5. **Dokumentation:** Algorithmus-MD mit Methodensatz, Schritten, Literatur, Verweis auf den Code.

Dann erscheint die Methode automatisch in der CLI (`--method`) und im Benchmark.

**Implementierung per KI/Assistent:** In `algorithms/Implementierungsvoraussetzungen.md` (Abschnitt 8) steht eine **Prompt-Vorlage**, die du kopieren und mit deinem Methodennamen sowie dem Pfad zu deiner Implementierungs-MD füllen kannst – so bekommst du eine Implementierung, die den Vertrag einhält.

---

## Python-API (ohne CLI)

```python
from pathlib import Path
from src.chunking import (
    run_full_analysis,   # Legacy: eine Datei, Community Network
    run_batch,           # Eine Methode, viele Dateien
    run_benchmark,       # Alle Methoden, gleiche Dateiliste
    load_srt_file,
    extract_ikis,
    list_methods,
)

# Einzeldatei (neue API)
from src.chunking import run_single_file, get_runner
result = run_single_file(
    "community_network",
    Path("SRT/Beispiel.csv"),
    Path("outputs/api_test"),
    sequence_type="blue",
    gamma=0.9,
    coupling=0.03,
)

# Batch
run_batch(
    method_name="community_network",
    input_dir="SRT",
    output_dir="outputs/batch",
    limit=5,
    random_state=42,
)
```

Notebooks können weiterhin `run_full_analysis` aus `src.chunking` verwenden (Legacy-API).

---

## Kurzüberblick

| Was du tun willst | Befehl / Ort |
|-------------------|--------------|
| Eine Datei analysieren | `python -m src.chunking --method community_network --input-file <path> --output-dir outputs` |
| Viele Dateien, eine Methode | `--input-dir SRT --output-dir outputs/batch` (evtl. `--limit 10`) |
| Alle Methoden vergleichen | `--benchmark --input-dir SRT --output-dir outputs/benchmark` |
| Ausgabe verstehen | Siehe Abschnitt „Ausgabe“ (summary.csv, trials.csv, meta.json) |
| Neue Methode einbauen | `algorithms/Implementierungsvoraussetzungen.md` + `methods/<name>.py` + Registry |
| Algorithmen & Literatur | `algorithms/README.md`, `algorithms/community_network.md`, … |
