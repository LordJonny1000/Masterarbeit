
import requests
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path


class MultilingualEmbeddingExtractor:
    def __init__(self, embeddings_path: str = "data/numberbatch-19.08.txt"):
        self.embeddings_path = Path(embeddings_path)
        self.conceptnet_base_url = "http://api.conceptnet.io"

    def get_translations(self, word: str, source_lang: str = "en") -> Dict[str, str]:
        url = f"{self.conceptnet_base_url}/query?node=/c/{source_lang}/{word}&rel=/r/Synonym"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as e:
            print(f"Error fetching translations: {e}")
            return {}

        translations = {source_lang: word}
        for edge in data.get('edges', []):
            end_node = edge.get('end', {})
            lang = end_node.get('language')
            if lang and lang != source_lang:
                translations[lang] = end_node.get('label', '')

        return translations

    def load_embeddings_for_words(self, translations: Dict[str, str]) -> Dict[Tuple[str, str], np.ndarray]:
        embeddings = {}
        translations_lower = {lang: word.lower() for lang, word in translations.items()}

        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_path}")

        with open(self.embeddings_path, 'r', encoding='utf-8') as f:
            next(f)

            for line in tqdm(f, desc="Searching embeddings", file=sys.stdout):
                parts = line.rstrip().split(' ')
                if len(parts) < 2:
                    continue

                entry = parts[0]
                entry_parts = entry.split('/')

                if len(entry_parts) >= 4:
                    target_lang = entry_parts[2]
                    target_word = entry_parts[3].lower()

                    if target_lang in translations_lower and translations_lower[target_lang] == target_word:
                        vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                        embeddings[(target_lang, translations[target_lang])] = vector

                        if len(embeddings) == len(translations):
                            break

        return embeddings

    def get_multilingual_embeddings(self, word: str, source_lang: str = "en") -> Dict[Tuple[str, str], np.ndarray]:
        translations = self.get_translations(word, source_lang)
        if not translations:
            return {}

        print(f"Found translations for '{word}': {translations}")
        embeddings = self.load_embeddings_for_words(translations)

        return embeddings


def analyze_embedding_consistency(embeddings: Dict[Tuple[str, str], np.ndarray]) -> Dict[str, float]:
    if len(embeddings) < 2:
        return {}

    vectors = list(embeddings.values())
    mean_vector = np.mean(vectors, axis=0)

    consistencies = {}
    for (lang, word), vector in embeddings.items():
        cosine_sim = np.dot(vector, mean_vector) / (np.linalg.norm(vector) * np.linalg.norm(mean_vector))
        consistencies[f"{lang}:{word}"] = float(cosine_sim)

    return consistencies


if __name__ == "__main__":
    extractor = MultilingualEmbeddingExtractor()

    test_word = "elephant"
    embeddings = extractor.get_multilingual_embeddings(test_word)

    print(f"\nFound {len(embeddings)} embeddings:")
    for (lang, word), vector in embeddings.items():
        print(f"  {lang}: {word} - Vector shape: {vector}")

