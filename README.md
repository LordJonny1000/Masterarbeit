# Multilingual Embedding Analysis - Befehle

## Extract - Embeddings extrahieren

```bash
# Standard: Alle Wörter aus concept_list.txt
python src/cli.py extract --batch

# Mit Cache (empfohlen)
python src/cli.py extract --batch --cache-dir data/cache

# Anhängen statt überschreiben
python src/cli.py extract --append --batch

# Eigene Wortliste
python src/cli.py extract --word-file data/my_words.txt --batch


# Andere Quellsprache
python src/cli.py extract --source-lang de --words Hund Katze --batch
```

## Manage - Daten verwalten

```bash
# Statistiken anzeigen
python src/cli.py manage --data-file data/processed/embeddings.h5 --stats

# Alle Konzepte auflisten
python src/cli.py manage --data-file data/processed/embeddings.h5 --list

# Detaillierte Liste mit Sprachen
python src/cli.py manage --data-file data/processed/embeddings.h5 --list --detailed

# Konzepte löschen
python src/cli.py manage --data-file data/processed/embeddings.h5 --delete word1 word2

# Nach NumPy exportieren
python src/cli.py manage --data-file data/processed/embeddings.h5 --export data/numpy_export
```

## Analyze - Analysen durchführen

```bash
# Konsistenz-Scores berechnen
python src/cli.py analyze --data-file data/processed/embeddings.h5 --consistency

# Sprachähnlichkeitsmatrix
python src/cli.py analyze --data-file data/processed/embeddings.h5 --language-similarity

# Outlier-Sprachen finden
python src/cli.py analyze --data-file data/processed/embeddings.h5 --outliers

# Top-20 wichtigste Dimensionen
python src/cli.py analyze --data-file data/processed/embeddings.h5 --dimensions 20

# Kompletter Report
python src/cli.py analyze --data-file data/processed/embeddings.h5 --report

# Visualisierung erstellen
python src/cli.py analyze --data-file data/processed/embeddings.h5 --visualize

# Alles mit Output-Verzeichnis
python src/cli.py analyze --data-file data/processed/embeddings.h5 \
    --consistency --language-similarity --outliers --dimensions 50 \
    --report --visualize --output-dir data/results

# Nur bestimmte Konzepte analysieren
python src/cli.py analyze --data-file data/processed/embeddings.h5 --concepts dog cat elephant --consistency
```
