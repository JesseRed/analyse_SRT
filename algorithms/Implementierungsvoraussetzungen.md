# Implementierungsvoraussetzungen für neue Chunking-Methoden

Diese Datei dient als **Checkliste und Vertrag** für jede neue Chunking-Methode. Beim Implementieren neuer Methoden die Punkte durchgehen und abhaken.

---

## 1. Eingabedaten

- [ ] **Gemeinsame Datenpipeline nutzen:** `load_srt_file` und `extract_ikis` aus `src.chunking._data` (bzw. `from src.chunking import load_srt_file, extract_ikis`). Keine abweichenden QC-Regeln ohne Begründung und Dokumentation.
- [ ] **Erwartetes Format:** SRT-CSV mit definierten Spalten: `BlockNumber`, `EventNumber`, `Time Since Block start`, `isHit`, `target`, `pressed`, `sequence`. 8-Item-Sequenzen; IKIs als Differenzen von „Time Since Block start“ (siehe `extract_ikis`).

---

## 2. Schnittstelle (Code-Contract)

- [ ] **Implementierung** in `src/chunking/methods/<method_name>.py` mit einer registrierten Funktion, die das gemeinsame **`ChunkingResult`-Schema** zurückgibt (Summary + Trials + optional Validation/Artefakte).
- [ ] **Methodenname:** kurz, lowercase, z. B. `community_network`, `change_point_pelt`.
- [ ] **Parameter:** Alle Laufparameter als Keyword-Argumente mit Defaults; Repräsentation in `parameters.json` muss JSON-serialisierbar sein.

---

## 3. Output-Pflichtfelder

- [ ] **Summary:** mindestens `source_file`, `sequence_type`, `method`, `n_blocks`. Empfohlen: eine Kennzahl „mittlere Chunk-Anzahl“ (`mean_n_chunks`) oder Äquivalent.
- [ ] **Trials:** mindestens `source_file`, `sequence_type`, `method`, `block_number`, `n_chunks`, `chunk_boundaries`. Format von `chunk_boundaries`: Liste von Grenzpositionen zwischen 1 und 7 (Position nach IKI-Index).
- [ ] **Fehler:** fehlgeschlagene Dateien werden in `errors.csv` mit Spalten `source_file`, `error` erfasst (vom Run-Layer geschrieben).

---

## 4. Dokumentation (Algorithmus-MD)

- [ ] **Eine Datei** `algorithms/<method_name>.md` (oder sprechender Name, z. B. `change_point_pelt.md`).
- [ ] **Inhalt:** kurzer Methodensatz (Ziel), Schritte der Pipeline (Datenvorbereitung, Modell, Validierung, Output-Metriken), Literaturzitate (APA + optional BibTeX), Verweis auf die Implementierung (`src/chunking/methods/<name>.py`).

---

## 5. Literatur & Zitierfähigkeit

- [ ] **Primärquelle(n)** der Methode und verwendete Software (z. B. ruptures, leidenalg) mit Version/Zitat in der Algorithmus-MD und optional in `meta.json` dokumentieren.

---

## 6. Tests & Validierung

- [ ] **Mindestens:** Lauf auf einer kleinen Testdatei (z. B. eine SRT-Datei aus dem Repo) ohne Crash; Ausgabe erfüllt Schema (Summary + Trials).
- [ ] **Optional:** Null-Modell oder Permutationstest; wenn vorhanden, in Algorithmus-MD und Output (z. B. `p_value_permutation`) dokumentieren.

---

## 7. Registrierung

- [ ] **Methode** in `src/chunking/methods/__init__.py` in der Methoden-Registry eintragen (`REGISTRY["method_name"] = run_analysis`), damit CLI und Benchmark sie automatisch anbieten.

---

## 8. Prompt-Vorlage für die Implementierung (KI/Assistent)

Wenn du eine neue Datei mit einem Implementierungsvorschlag (z. B. `algorithms/meine_methode.md`) hast und die Implementierung von einem Assistenten erledigen lassen willst, kannst du folgenden **Prompt** verwenden (unten die Platzhalter ersetzen):

---

**Kopie-Vorlage (Platzhalter anpassen):**

```
Implementiere eine neue Chunking-Methode für das analyse_SRT-Projekt.

- **Methodenname:** [z. B. change_point_pelt]
- **Implementierungsvorschlag:** Siehe [algorithms/change_point_pelt.md] (oder Pfad zu deiner MD-Datei).

Vorgaben:
1. Vertrag einhalten: algorithms/Implementierungsvoraussetzungen.md vollständig beachten.
2. Eingabe: load_srt_file und extract_ikis aus src.chunking._data verwenden (gleiche IKIs wie alle Methoden).
3. Ausgabe: run_analysis(filepath, sequence_type, **kwargs) -> ChunkingResult. Summary- und Trials-Spalten wie in den Implementierungsvoraussetzungen (mindestens source_file, sequence_type, method, n_blocks, block_number, n_chunks, chunk_boundaries). chunk_boundaries: Liste von Grenzpositionen 1–7.
4. Code: src/chunking/methods/[method_name].py anlegen. Orientierung: src/chunking/methods/community_network.py (run_analysis, ChunkingResult bauen, parameters dict, optional validation).
5. Registrierung: In src/chunking/methods/__init__.py die Methode in REGISTRY eintragen.
6. CLI: Falls die Methode neue Parameter braucht (z. B. penalty, Kmax), in src/chunking/__main__.py Argumente hinzufügen und in method_params übergeben.
7. Abhängigkeiten: Neue Pakete in requirements.txt eintragen (z. B. ruptures für PELT).
8. Algorithmus-MD: Wenn noch nicht vorhanden, algorithms/[method_name].md aus dem Implementierungsvorschlag anlegen oder bestehende Datei mit Verweis auf die Implementierung ergänzen.

Bitte die Implementierung ausführen und mit einer Testdatei aus SRT/ (z. B. --limit 1) prüfen.
```

---

*Referenz: Plan „Chunking-Methoden: Vergleichsarchitektur und Output-Struktur“.*
