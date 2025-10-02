#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
import json
from typing import List
import pandas as pd
from embedding_extractor import EmbeddingExtractor
from data_manager import DataManager
from analyzer import EmbeddingAnalyzer


def extract_embeddings(args):
    base_dir = Path(__file__).parent.parent
    default_embeddings_file = base_dir / "data" / "numberbatch-19.08.txt"
    default_output_file = base_dir / "data" / "processed" / "embeddings.h5"
    default_word_file = base_dir / "data" / "concept_list.csv"

    embeddings_file = Path(args.embeddings_file) if args.embeddings_file else default_embeddings_file
    output_file = Path(args.output) if args.output else default_output_file

    extractor = EmbeddingExtractor(
        embeddings_file=embeddings_file,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None
    )

    words = []
    if args.words:
        words = args.words
        concept_properties = {word: {} for word in words}
    elif args.word_file:
        df = pd.read_csv(args.word_file)
        words = df['concept'].tolist()
        concept_properties = {}
        for _, row in df.iterrows():
            props = row.drop('concept').to_dict()
            for key, value in props.items():
                if pd.isna(value):
                    props[key] = None
                else:
                    try:
                        props[key] = int(value)
                    except ValueError:
                        try:
                            props[key] = float(value)
                        except ValueError:
                            props[key] = value
            concept_properties[row['concept']] = props
    else:
        if default_word_file.exists():
            df = pd.read_csv(default_word_file)
            words = df['concept'].tolist()
            concept_properties = {}
            for _, row in df.iterrows():
                props = row.drop('concept').to_dict()
                for key, value in props.items():
                    if pd.isna(value):
                        props[key] = None
                    else:
                        try:
                            props[key] = int(value)
                        except ValueError:
                            try:
                                props[key] = float(value)
                            except ValueError:
                                props[key] = value
                concept_properties[row['concept']] = props
        else:
            print(f"Error: No word list found at {default_word_file}")
            print("Provide either --words or --word-file, or create data/concept_list.csv")
            sys.exit(1)

    print(f"Extracting embeddings for {len(words)} concepts...")
    print(f"Source: {embeddings_file}")
    print(f"Output: {output_file}")

    if args.batch:
        concepts = extractor.extract_concepts_batch(
            words,
            args.source_lang,
            concept_properties=concept_properties
        )
    else:
        concepts = {}
        for word in words:
            print(f"Processing: {word}")
            concept = extractor.extract_concept(
                word,
                args.source_lang,
                properties=concept_properties.get(word, {})
            )
            concepts[word] = concept

    manager = DataManager(output_file)
    mode = 'a' if args.append else 'w'
    manager.save_concepts(concepts, mode=mode)

    print(f"\nExtraction complete! Summary:")
    for word, concept in concepts.items():
        print(f"  {word}: {len(concept.embeddings)} languages")


def analyze_embeddings(args):
    manager = DataManager(Path(args.data_file))
    analyzer = EmbeddingAnalyzer(
        output_dir=Path(args.output_dir) if args.output_dir else None
    )

    concept_ids = None
    if args.concepts:
        concept_ids = args.concepts

    print("Loading concepts...")
    concepts = manager.load_concepts(concept_ids)
    print(f"Loaded {len(concepts)} concepts")

    if args.consistency:
        print("\nAnalyzing consistency...")
        df = analyzer.compute_all_consistencies(concepts)
        print(df.to_string())
        if args.output_dir:
            df.to_csv(Path(args.output_dir) / "consistency.csv", index=False)

    if args.language_similarity:
        print("\nComputing language similarity matrix...")
        df = analyzer.compute_language_similarity_matrix(concepts)
        print(df.to_string())
        if args.output_dir:
            df.to_csv(Path(args.output_dir) / "language_similarity.csv")

    if args.outliers:
        print("\nFinding outlier languages...")
        outliers = analyzer.find_outlier_languages(concepts, threshold_std=args.outlier_threshold)
        for concept_id, langs in outliers.items():
            print(f"{concept_id}: {', '.join(langs)}")

    if args.dimensions:
        print(f"\nAnalyzing top {args.dimensions} dimensions...")
        df = analyzer.analyze_dimension_importance(concepts, top_k=args.dimensions)
        print(df.to_string())
        if args.output_dir:
            df.to_csv(Path(args.output_dir) / "dimension_importance.csv", index=False)

    if args.report:
        print("\nGenerating report...")
        report = analyzer.generate_report(concepts)
        print(report)
        if args.output_dir:
            with open(Path(args.output_dir) / "report.txt", 'w') as f:
                f.write(report)

    if args.visualize:
        print("\nCreating visualization...")
        save_path = Path(args.output_dir) / "language_similarity.png" if args.output_dir else None
        analyzer.visualize_language_similarities(concepts, save_path)


def manage_data(args):
    manager = DataManager(Path(args.data_file))

    if args.stats:
        stats = manager.get_statistics()
        print(json.dumps(stats, indent=2, default=str))

    elif args.list:
        concepts = manager.list_concepts()
        if args.detailed:
            loaded_concepts = manager.load_concepts()
            for concept_id, concept in loaded_concepts.items():
                langs = list(concept.embeddings.keys())
                print(f"{concept_id}: {len(langs)} languages ({', '.join(langs[:5])}{'...' if len(langs) > 5 else ''})")
        else:
            for concept in concepts:
                print(concept)

    elif args.delete:
        count = manager.delete_concepts(args.delete)
        print(f"Deleted {count} concepts")

    elif args.export:
        output_dir = Path(args.export)
        concept_ids = args.concepts if args.concepts else None
        manager.export_to_numpy(output_dir, concept_ids)


def probe_semantic(args):
    print("Semantic probing functionality will be implemented here")
    print("This will train models to predict semantic properties from embeddings")


def main():
    parser = argparse.ArgumentParser(
        description="Multilingual Embedding Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    extract_parser = subparsers.add_parser('extract', help='Extract embeddings for concepts')
    extract_parser.add_argument('--embeddings-file',
                                help='Path to ConceptNet embeddings file (default: data/numberbatch-19.08.txt)')
    extract_parser.add_argument('--output', help='Output HDF5 file (default: data/processed/embeddings.h5)')
    extract_parser.add_argument('--words', nargs='+', help='List of words to extract (overrides word file)')
    extract_parser.add_argument('--word-file', help='File containing words (default: data/concept_list.csv)')
    extract_parser.add_argument('--source-lang', default='en', help='Source language (default: en)')
    extract_parser.add_argument('--append', action='store_true', help='Append to existing file')
    extract_parser.add_argument('--batch', action='store_true', help='Use batch extraction (faster)')
    extract_parser.add_argument('--cache-dir', help='Directory for caching translations')

    analyze_parser = subparsers.add_parser('analyze', help='Analyze extracted embeddings')
    analyze_parser.add_argument('--data-file', required=True, help='Input HDF5 file')
    analyze_parser.add_argument('--concepts', nargs='+', help='Specific concepts to analyze')
    analyze_parser.add_argument('--consistency', action='store_true', help='Compute consistency scores')
    analyze_parser.add_argument('--language-similarity', action='store_true', help='Compute language similarity matrix')
    analyze_parser.add_argument('--outliers', action='store_true', help='Find outlier languages')
    analyze_parser.add_argument('--outlier-threshold', type=float, default=2.0,
                                help='Outlier threshold in std deviations')
    analyze_parser.add_argument('--dimensions', type=int, help='Analyze top N important dimensions')
    analyze_parser.add_argument('--report', action='store_true', help='Generate comprehensive report')
    analyze_parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    analyze_parser.add_argument('--output-dir', help='Directory for output files')

    manage_parser = subparsers.add_parser('manage', help='Manage HDF5 data files')
    manage_parser.add_argument('--data-file', required=True, help='HDF5 data file')
    manage_parser.add_argument('--stats', action='store_true', help='Show statistics')
    manage_parser.add_argument('--list', action='store_true', help='List all concepts')
    manage_parser.add_argument('--detailed', action='store_true', help='Show detailed listing')
    manage_parser.add_argument('--delete', nargs='+', help='Delete specified concepts')
    manage_parser.add_argument('--export', help='Export to numpy format')
    manage_parser.add_argument('--concepts', nargs='+', help='Specific concepts for export')

    probe_parser = subparsers.add_parser('probe', help='Probe for semantic properties')
    probe_parser.add_argument('--data-file', required=True, help='Input HDF5 file')
    probe_parser.add_argument('--property', required=True, choices=['animate', 'concrete', 'countable'],
                              help='Semantic property to probe')
    probe_parser.add_argument('--annotations', help='JSON file with property annotations')
    probe_parser.add_argument('--output-dir', help='Directory for results')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == 'extract':
        extract_embeddings(args)
    elif args.command == 'analyze':
        analyze_embeddings(args)
    elif args.command == 'manage':
        manage_data(args)
    elif args.command == 'probe':
        probe_semantic(args)


if __name__ == "__main__":
    main()



    #
    # manager = DataManager(Path("data/processed/embeddings.h5"))
    # print("Number of concepts: ", len(manager.list_concepts()))
    # concepts = manager.load_concepts()
    #
    # from sklearn.model_selection import train_test_split
    # from sklearn.linear_model import LogisticRegression
    # from sklearn.metrics import accuracy_score
    # import numpy as np
    # import copy as cp
    #
    #
    #
    #
    # X = np.empty((0, 300))
    # y = np.empty((0))
    #
    #
    #
    # for concept in concepts:
    #     embeddings = np.stack([x for x in concepts[concept].embeddings.values()])
    #     X = np.vstack([X, embeddings])
    #     labels = np.full(embeddings.shape[0], list(concepts[concept].properties.values())[0]) #material,countable,living
    #     y = np.concatenate((y, labels)) # hier eigenschaft aussuchen
    #
    #
    # n_dims = X.shape[1]
    #
    # X_clone = cp.copy(X)
    #
    #
    #
    # for n in range(n_dims):
    #
    #     X = X_clone[:, n].reshape(-1, 1)
    #
    #
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #
    #     clf = LogisticRegression()
    #     clf.fit(X_train, y_train)
    #
    #     y_pred = clf.predict(X_test)
    #     accuracy = accuracy_score(y_test, y_pred)
    #
    #     print(f"Accuracy only with dimension {n}: {accuracy:.2f}")
