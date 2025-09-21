import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import procrustes
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from embedding_extractor import Concept


class EmbeddingAnalyzer:
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_concept_consistency(self, concept: Concept) -> Dict:
        if len(concept.embeddings) < 2:
            return {'error': 'Need at least 2 languages for consistency analysis'}

        vectors = list(concept.embeddings.values())
        mean_vector = np.mean(vectors, axis=0)

        similarities = []
        for lang, vector in concept.embeddings.items():
            cos_sim = cosine_similarity(
                vector.reshape(1, -1),
                mean_vector.reshape(1, -1)
            )[0, 0]
            similarities.append(cos_sim)

        return {
            'mean_similarity': float(np.mean(similarities)),
            'std_similarity': float(np.std(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'range_similarity': float(np.max(similarities) - np.min(similarities)),
            'num_languages': len(concept.embeddings),
            'languages': list(concept.embeddings.keys())
        }

    def compute_all_consistencies(self, concepts: Dict[str, Concept]) -> pd.DataFrame:
        results = []

        for concept_id, concept in concepts.items():
            consistency = self.compute_concept_consistency(concept)
            if 'error' not in consistency:
                consistency['concept_id'] = concept_id
                results.append(consistency)

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('mean_similarity', ascending=False)

        return df

    def compute_language_similarity_matrix(self, concepts: Dict[str, Concept],
                                           min_shared_concepts: int = 5) -> pd.DataFrame:
        all_langs = set()
        for concept in concepts.values():
            all_langs.update(concept.embeddings.keys())

        all_langs = sorted(list(all_langs))
        n_langs = len(all_langs)

        similarity_matrix = np.zeros((n_langs, n_langs))
        count_matrix = np.zeros((n_langs, n_langs))

        for concept in concepts.values():
            for i, lang1 in enumerate(all_langs):
                for j, lang2 in enumerate(all_langs):
                    if lang1 in concept.embeddings and lang2 in concept.embeddings:
                        vec1 = concept.embeddings[lang1]
                        vec2 = concept.embeddings[lang2]
                        sim = cosine_similarity(
                            vec1.reshape(1, -1),
                            vec2.reshape(1, -1)
                        )[0, 0]
                        similarity_matrix[i, j] += sim
                        count_matrix[i, j] += 1

        with np.errstate(divide='ignore', invalid='ignore'):
            avg_similarity = np.where(
                count_matrix >= min_shared_concepts,
                similarity_matrix / count_matrix,
                np.nan
            )

        df = pd.DataFrame(avg_similarity, index=all_langs, columns=all_langs)
        return df

    def compute_pairwise_concept_similarity(self, concept1: Concept, concept2: Concept) -> pd.DataFrame:
        common_langs = set(concept1.embeddings.keys()) & set(concept2.embeddings.keys())

        if not common_langs:
            return pd.DataFrame()

        results = []
        for lang in common_langs:
            vec1 = concept1.embeddings[lang]
            vec2 = concept2.embeddings[lang]
            sim = cosine_similarity(
                vec1.reshape(1, -1),
                vec2.reshape(1, -1)
            )[0, 0]
            results.append({
                'language': lang,
                'similarity': sim,
                'word1': concept1.translations.get(lang, ''),
                'word2': concept2.translations.get(lang, '')
            })

        df = pd.DataFrame(results)
        df = df.sort_values('similarity', ascending=False)
        return df

    def find_outlier_languages(self, concepts: Dict[str, Concept],
                               threshold_std: float = 2.0) -> Dict[str, List[str]]:
        outliers = {}

        for concept_id, concept in concepts.items():
            if len(concept.embeddings) < 3:
                continue

            vectors = []
            langs = []
            for lang, vec in concept.embeddings.items():
                vectors.append(vec)
                langs.append(lang)

            mean_vector = np.mean(vectors, axis=0)

            similarities = []
            for vec in vectors:
                sim = cosine_similarity(
                    vec.reshape(1, -1),
                    mean_vector.reshape(1, -1)
                )[0, 0]
                similarities.append(sim)

            mean_sim = np.mean(similarities)
            std_sim = np.std(similarities)

            outlier_langs = []
            for i, (lang, sim) in enumerate(zip(langs, similarities)):
                if abs(sim - mean_sim) > threshold_std * std_sim:
                    outlier_langs.append(lang)

            if outlier_langs:
                outliers[concept_id] = outlier_langs

        return outliers

    def visualize_language_similarities(self, concepts: Dict[str, Concept],
                                        save_path: Optional[Path] = None):
        similarity_matrix = self.compute_language_similarity_matrix(concepts)

        plt.figure(figsize=(12, 10))
        mask = similarity_matrix.isna()

        sns.heatmap(similarity_matrix,
                    annot=True,
                    fmt='.2f',
                    cmap='coolwarm',
                    center=0.5,
                    vmin=0, vmax=1,
                    mask=mask,
                    square=True,
                    cbar_kws={"shrink": .8})

        plt.title('Cross-lingual Embedding Similarities', fontsize=16)
        plt.xlabel('Language', fontsize=12)
        plt.ylabel('Language', fontsize=12)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()

        return similarity_matrix

    def analyze_dimension_importance(self, concepts: Dict[str, Concept],
                                     top_k: int = 20) -> pd.DataFrame:
        all_embeddings = []

        for concept in concepts.values():
            for embedding in concept.embeddings.values():
                all_embeddings.append(embedding)

        if not all_embeddings:
            return pd.DataFrame()

        embeddings_matrix = np.array(all_embeddings)

        variances = np.var(embeddings_matrix, axis=0)
        means = np.mean(np.abs(embeddings_matrix), axis=0)

        dim_stats = []
        for i in range(len(variances)):
            dim_stats.append({
                'dimension': i,
                'variance': variances[i],
                'mean_abs_value': means[i],
                'importance_score': variances[i] * means[i]
            })

        df = pd.DataFrame(dim_stats)
        df = df.sort_values('importance_score', ascending=False)

        return df.head(top_k)

    def compute_semantic_field_structure(self, concepts: Dict[str, Concept],
                                         language: str = 'en') -> pd.DataFrame:
        concept_vectors = {}

        for concept_id, concept in concepts.items():
            if language in concept.embeddings:
                concept_vectors[concept_id] = concept.embeddings[language]

        if len(concept_vectors) < 2:
            return pd.DataFrame()

        concept_ids = list(concept_vectors.keys())
        n = len(concept_ids)
        similarity_matrix = np.zeros((n, n))

        for i, id1 in enumerate(concept_ids):
            for j, id2 in enumerate(concept_ids):
                vec1 = concept_vectors[id1]
                vec2 = concept_vectors[id2]
                similarity_matrix[i, j] = cosine_similarity(
                    vec1.reshape(1, -1),
                    vec2.reshape(1, -1)
                )[0, 0]

        df = pd.DataFrame(similarity_matrix,
                          index=concept_ids,
                          columns=concept_ids)

        return df

    def generate_report(self, concepts: Dict[str, Concept]) -> str:
        report = []
        report.append("=" * 60)
        report.append("MULTILINGUAL EMBEDDING ANALYSIS REPORT")
        report.append("=" * 60)

        report.append(f"\nTotal concepts analyzed: {len(concepts)}")

        all_langs = set()
        for concept in concepts.values():
            all_langs.update(concept.embeddings.keys())
        report.append(f"Total languages: {len(all_langs)}")
        report.append(f"Languages: {', '.join(sorted(all_langs))}")

        consistency_df = self.compute_all_consistencies(concepts)
        if not consistency_df.empty:
            report.append("\n" + "-" * 40)
            report.append("TOP 5 MOST CONSISTENT CONCEPTS:")
            report.append("-" * 40)
            for _, row in consistency_df.head(5).iterrows():
                report.append(f"{row['concept_id']}: {row['mean_similarity']:.3f} "
                              f"(±{row['std_similarity']:.3f})")

            report.append("\n" + "-" * 40)
            report.append("TOP 5 LEAST CONSISTENT CONCEPTS:")
            report.append("-" * 40)
            for _, row in consistency_df.tail(5).iterrows():
                report.append(f"{row['concept_id']}: {row['mean_similarity']:.3f} "
                              f"(±{row['std_similarity']:.3f})")

        outliers = self.find_outlier_languages(concepts)
        if outliers:
            report.append("\n" + "-" * 40)
            report.append("CONCEPTS WITH OUTLIER LANGUAGES:")
            report.append("-" * 40)
            for concept_id, langs in list(outliers.items())[:5]:
                report.append(f"{concept_id}: {', '.join(langs)}")

        lang_sim_matrix = self.compute_language_similarity_matrix(concepts)
        if not lang_sim_matrix.empty:
            mean_similarities = lang_sim_matrix.mean(axis=1).sort_values(ascending=False)
            report.append("\n" + "-" * 40)
            report.append("LANGUAGE SIMILARITY RANKINGS:")
            report.append("-" * 40)
            for lang, sim in mean_similarities.head(10).items():
                if not np.isnan(sim):
                    report.append(f"{lang}: {sim:.3f}")

        return "\n".join(report)