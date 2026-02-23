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
├── infos.md                  # Kontext: SRTT, 60/30/30-Design, Literatur
│
├── algorithms/               # Nur Dokumentation (kein Code)
│   ├── README.md             # Übersicht aller Chunking-Methoden
│   ├── Implementierungsvoraussetzungen.md   # Checkliste für neue Methoden
│   ├── community_network.md  # Wymbs/Mucha – Implementierungsplan + Literatur
│   └── change_point_pelt.md   # PELT/ruptures – Methode + Literatur
│
├── src/
│   └── chunking/             # Chunking-Paket
│       ├── __init__.py        # Re-Exports, Legacy-API
│       ├── __main__.py        # CLI (--method, --benchmark, …)
│       ├── _base.py           # ChunkingResult, Schema
│       ├── _data.py           # load_srt_file, extract_ikis (gemeinsam)
│       ├── _compat.py         # run_full_analysis, run_batch_analysis (Legacy)
│       ├── run.py             # run_single_file, run_batch, run_benchmark
│       ├── benchmark_eval.py  # Auswertung: Vergleichstabelle, ARI
│       └── methods/
│           ├── __init__.py    # Methoden-Registry
│           ├── community_network.py
│           └── change_point_pelt.py
│
├── SRT/                      # Eingabe: Teilnehmer-CSVs (z. B. *.csv)
├── outputs/                  # Ausgabe pro Lauf (siehe unten)
└── notebooks/                # z. B. chunking_analysis.ipynb
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

Methodenspezifische Optionen (z. B. für Community Network):

- `--gamma 0.9` (Intralayer-Auflösung)  
- `--coupling 0.03` (Interlayer-Kopplung)  
- `--n-iter 20` (Wiederholungen pro Datei)  
- `--n-permutations 20` (Nullmodell-Permutationen)  
- `--seed 42` (Reproduzierbarkeit)

Für **change_point_pelt**: `--penalty-sensitive`, `--short-session`, `--cost-model l2|rbf`, `--rbf-gamma`, `--window-size`, `--step`, `--min-blocks` (siehe `algorithms/change_point_pelt.md`).

### 3. Benchmark: alle Methoden auf denselben Dateien

```bash
python -m src.chunking \
  --benchmark \
  --input-dir SRT \
  --output-dir outputs/benchmark \
  --limit 5
```

- Pro registrierte Methode wird ein Unterordner angelegt (`outputs/benchmark/community_network/`, …).  
- Zusätzlich: `outputs/benchmark/benchmark_summary.csv` – eine Zeile pro **Datei × Methode** (Long-Format) für direkte Vergleiche.

### 4. Verfügbare Methoden anzeigen

Aktuell sind **community_network** und **change_point_pelt** implementiert; die CLI zeigt die Liste der Methoden über `--method` mit an (z. B. in der Fehlermeldung bei falschem Namen).

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
- **Empfohlen/Gemeinsam:** `mean_n_chunks`  
- **Run-Layer-Metadaten (optional):** `participant_id`, `session`, `day_index` (aus `source_file` abgeleitet)
- **Methodenspezifisch** (z. B. Community Network): `mean_q_single_trial`, `mean_phi`, `empirical_q_multitrial`, `p_value_permutation`, …

Eine Zeile = eine analysierte Teilnehmerdatei. Ideal für übergreifende Auswertungen (z. B. nach Gruppe, Tag, Bedingung).

### trials.csv

- **Mindestspalten:** `source_file`, `sequence_type`, `method`, `block_number`, `n_chunks`, `chunk_boundaries`  
- **chunk_boundaries:** Liste von Grenzpositionen (1–7), einheitliches Format über alle Methoden  
- **Run-Layer-Metadaten (optional):** `participant_id`, `session`, `day_index`  
- Zusätzliche Spalten je Methode (z. B. `q_single_trial`, `phi`, `community_labels`, `ikis`)

Eine Zeile = ein Block/Trial. Für trialweise Auswertungen und Methodenvergleiche (z. B. ARI).

### meta.json / parameters.json

- **Reproduzierbarkeit:** Zeitstempel, optional Git-Commit, Eingabe-Pfad/Pattern, alle Parameter.  
- Für Methodenpapiere und Supplement: Welche Softwareversion, welche Einstellungen.

### validation.json / artifacts (optional)

- Falls eine Methode `result.validation` liefert, wird sie als JSON persistiert:
  - Single-File: `validation.json` im Methoden-Output.
  - Batch: pro Quelldatei unter `artifacts/<source_stem>/validation.json`.

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
    method_b="change_point_pelt",  # sobald implementiert
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
