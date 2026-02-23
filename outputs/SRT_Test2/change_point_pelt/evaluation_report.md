# Auswertung: Chunking über Tage und Sequenzen

Dieser Bericht fasst die Change-Point-basierte Chunk-Analyse (PELT) für die Sequenztypen Blue, Green und Yellow zusammen. Untersucht werden (1) die Veränderung des Chunkings über die Messzeitpunkte (Tage) pro Sequenz und Proband sowie (2) die Unterschiede zwischen den Sequenzen (strukturierte Sequenzen Blue/Green vs. Null-Bedingung Yellow).


## 1. Veränderung des Chunkings über die Tage

Für jede Sequenz (Blue, Green, Yellow) und jeden Probanden wird die mittlere Chunk-Anzahl (*mean_n_chunks*) über die Messzeitpunkte (Tag 1, 2, 3) beschrieben. *mean_n_chunks* > 1 deutet auf erkennbare Chunk-Grenzen hin.

### Sequenz: Blue

- **Egbert_Hausmann**: Tag 1: 1.00, Tag 2: 1.00, Tag 3: 1.21
- **Fritz_Raschke**: Tag 1: 1.19, Tag 2: 1.00, Tag 3: 1.00
- **Liane_Rieger**: Tag 2: 1.00, Tag 3: 1.00

### Sequenz: Green

- **Egbert_Hausmann**: Tag 1: 1.00, Tag 2: 1.04, Tag 3: 1.00
- **Fritz_Raschke**: Tag 1: 1.60, Tag 2: 1.00, Tag 3: 1.00
- **Liane_Rieger**: Tag 2: 1.00

### Sequenz: Yellow

- **Egbert_Hausmann**: Tag 1: 1.00, Tag 2: 1.00, Tag 3: 1.10
- **Fritz_Raschke**: Tag 1: 1.00, Tag 2: 1.09, Tag 3: 1.00
- **Liane_Rieger**: Tag 1: 1.00, Tag 2: 1.00, Tag 3: 1.00

### Hinweise zu Datenqualität

Folgende Messzeitpunkte haben weniger als 3 gültige Fenster; die Ergebnisse sind mit Vorsicht zu interpretieren:

- **Liane_Rieger**, Sequenz Blue, Tag 3: nur 1 gültige Fenster (Ergebnisse mit Vorsicht interpretieren)
- **Liane_Rieger**, Sequenz Yellow, Tag 3: nur 2 gültige Fenster (Ergebnisse mit Vorsicht interpretieren)
- **Liane_Rieger**, Sequenz Yellow, Tag 1: nur 2 gültige Fenster (Ergebnisse mit Vorsicht interpretieren)

### Kurzfassung Verlauf

- **Blue**: Mittelwerte (M, Min, Max, SD) pro Proband: 
                  mean  min    max    std
participant_id                           
Egbert_Hausmann  1.069  1.0  1.208  0.120
Fritz_Raschke    1.065  1.0  1.194  0.112
Liane_Rieger     1.000  1.0  1.000  0.000

- **Green**: Mittelwerte (M, Min, Max, SD) pro Proband: 
                  mean  min    max    std
participant_id                           
Egbert_Hausmann  1.014  1.0  1.043  0.025
Fritz_Raschke    1.200  1.0  1.600  0.346
Liane_Rieger     1.000  1.0  1.000    NaN

- **Yellow**: Mittelwerte (M, Min, Max, SD) pro Proband: 
                  mean  min    max    std
participant_id                           
Egbert_Hausmann  1.034  1.0  1.103  0.060
Fritz_Raschke    1.030  1.0  1.091  0.052
Liane_Rieger     1.000  1.0  1.000  0.000


## 2. Unterschiede zwischen den Sequenzen

Yellow dient als Null-Bedingung (zufällige Sequenz); Blue und Green sind strukturierte Sequenzen. Positive Deltas (Blue/Green minus Yellow) bedeuten mehr Chunk-Struktur in der strukturierten Sequenz.

### Strukturierte Sequenzen vs. Yellow (pro Messzeitpunkt)

**Egbert_Hausmann**, Tag 1: Blue−Yellow = 0.000, Green−Yellow = 0.000
**Egbert_Hausmann**, Tag 2: Blue−Yellow = 0.000, Green−Yellow = 0.043
**Egbert_Hausmann**, Tag 3: Blue−Yellow = 0.105, Green−Yellow = -0.103
**Fritz_Raschke**, Tag 1: Blue−Yellow = 0.194, Green−Yellow = 0.600
**Fritz_Raschke**, Tag 2: Blue−Yellow = -0.091, Green−Yellow = -0.091
**Fritz_Raschke**, Tag 3: Blue−Yellow = 0.000, Green−Yellow = 0.000
**Liane_Rieger**, Tag 2: Blue−Yellow = 0.000, Green−Yellow = 0.000
**Liane_Rieger**, Tag 3: Blue−Yellow = 0.000

### Aggregation über Probanden und Tage

Sequenz  M (mean_n_chunks)     SD  Min    Max  N
   Blue             1.0502 0.0931  1.0 1.2083  8
  Green             1.0919 0.2246  1.0 1.6000  7
 Yellow             1.0216 0.0430  1.0 1.1034  9

Deltas (strukturiert − Yellow):
- Blue−Yellow: M = 0.026, SD = 0.086
- Green−Yellow: M = 0.064, SD = 0.242
