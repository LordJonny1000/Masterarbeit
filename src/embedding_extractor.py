import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import requests
import time
import json
from tqdm import tqdm
from dataclasses import dataclass
from deep_translator import GoogleTranslator


@dataclass
class Concept:
    concept_id: str
    translations: Dict[str, str]
    embeddings: Dict[str, np.ndarray]
    properties: Dict[str, any]
    metadata: Dict[str, any]


class EmbeddingExtractor:
    def __init__(self, embeddings_file: Path, cache_dir: Optional[Path] = None,
                 use_google_fallback: bool = True):
        self.embeddings_file = Path(embeddings_file)
        if not self.embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_file}")

        self.cache_dir = cache_dir
        self.use_google_fallback = use_google_fallback

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.translation_cache_file = self.cache_dir / "translation_cache.json"
            self.translation_cache = self._load_translation_cache()
        else:
            self.translation_cache = {}

        self.embeddings_index = None
        self._build_index()

    def _build_index(self):
        print("Building embeddings index (this may take a while on first run)...")
        index_file = self.embeddings_file.parent / f"{self.embeddings_file.stem}_index.json"

        if index_file.exists():
            print(f"Loading index from {index_file}")
            with open(index_file, 'r') as f:
                self.embeddings_index = json.load(f)
        else:
            self.embeddings_index = {}
            with open(self.embeddings_file, 'r', encoding='utf-8') as f:
                next(f)
                for i, line in enumerate(tqdm(f, desc="Indexing embeddings")):
                    if i % 100000 == 0:
                        parts = line.split(' ', 1)
                        if len(parts) >= 1:
                            entry = parts[0]
                            entry_parts = entry.split('/')
                            if len(entry_parts) >= 4:
                                lang = entry_parts[2]
                                word = entry_parts[3].lower()
                                key = f"{lang}:{word}"
                                self.embeddings_index[key] = i

            with open(index_file, 'w') as f:
                json.dump(self.embeddings_index, f)
            print(f"Saved index to {index_file}")

    def _load_translation_cache(self) -> Dict:
        if self.translation_cache_file.exists():
            with open(self.translation_cache_file, 'r') as f:
                return json.load(f)
        return {}

    def _save_translation_cache(self):
        if self.cache_dir and self.translation_cache:
            with open(self.translation_cache_file, 'w') as f:
                json.dump(self.translation_cache, f, ensure_ascii=False, indent=2)

    def get_translations_google(self, word: str, source_lang: str = 'en',
                                target_langs: List[str] = None) -> Dict[str, str]:
        if target_langs is None:
            target_langs = ['de', 'fr', 'es', 'it', 'pt', 'nl', 'pl', 'ru',
                            'ja', 'ko', 'zh-CN', 'ar', 'hi', 'tr', 'sv']

        cache_key = f"google:{source_lang}:{word}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]

        translations = {source_lang: word}

        for lang in target_langs:
            try:
                translator = GoogleTranslator(source=source_lang, target=lang)
                result = translator.translate(word)
                if result:
                    translations[lang] = result.lower()
                time.sleep(0.05)
            except Exception as e:
                continue

        self.translation_cache[cache_key] = translations
        self._save_translation_cache()
        return translations

    def get_translations_conceptnet(self, word: str, source_lang: str = 'en',
                                    max_retries: int = 3) -> Dict[str, str]:
        cache_key = f"conceptnet:{source_lang}:{word}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]

        translations = {source_lang: word}

        for rel in ['/r/Synonym', '/r/TranslatedAs']:
            for attempt in range(max_retries):
                try:
                    url = f"http://api.conceptnet.io/query?node=/c/{source_lang}/{word}&rel={rel}"
                    response = requests.get(url, timeout=5)

                    if response.status_code == 502:
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)
                            continue
                        else:
                            raise Exception("ConceptNet API unavailable")

                    response.raise_for_status()
                    data = response.json()

                    for edge in data.get('edges', []):
                        for node_key in ['end', 'start']:
                            node = edge.get(node_key, {})
                            if 'language' in node and 'label' in node:
                                lang = node['language']
                                label = node['label'].lower()
                                if lang not in translations:
                                    translations[lang] = label

                    time.sleep(0.3)
                    break

                except Exception as e:
                    if attempt == max_retries - 1:
                        if self.use_google_fallback:
                            return self.get_translations_google(word, source_lang)
                        else:
                            print(f"Failed to get translations for '{word}': {e}")
                            return {source_lang: word}

        self.translation_cache[cache_key] = translations
        self._save_translation_cache()
        return translations

    def get_translations_manual(self, word: str, translations: Dict[str, str]) -> Dict[str, str]:
        cache_key = f"manual:{word}"
        self.translation_cache[cache_key] = translations
        self._save_translation_cache()
        return translations

    def extract_embedding(self, lang: str, word: str) -> Optional[np.ndarray]:
        lang_mapping = {
            'zh-CN': 'zh',
            'zh-cn': 'zh',
            'zh': 'zh'
        }

        lang = lang_mapping.get(lang, lang)

        with open(self.embeddings_file, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                parts = line.rstrip().split(' ')
                if len(parts) < 2:
                    continue

                entry = parts[0]
                entry_parts = entry.split('/')

                if len(entry_parts) >= 4:
                    entry_lang = entry_parts[2]
                    entry_word = entry_parts[3].lower()

                    if entry_lang == lang and entry_word == word.lower():
                        return np.array([float(x) for x in parts[1:]], dtype=np.float32)
        return None

    def extract_concept(self, word: str, source_lang: str = 'en',
                        manual_translations: Optional[Dict[str, str]] = None) -> Concept:
        if manual_translations:
            translations = self.get_translations_manual(word, manual_translations)
        else:
            translations = self.get_translations_conceptnet(word, source_lang)

        embeddings = {}
        for lang, translated_word in translations.items():
            embedding = self.extract_embedding(lang, translated_word)
            if embedding is not None:
                embeddings[lang] = embedding

        return Concept(
            concept_id=word,
            translations=translations,
            embeddings=embeddings,
            properties={},
            metadata={'source_lang': source_lang,
                      'extraction_method': 'manual' if manual_translations else 'auto'}
        )

    def extract_concepts_batch(self, words: List[str], source_lang: str = 'en',
                               show_progress: bool = True) -> Dict[str, Concept]:
        concepts = {}

        all_translations = {}
        for word in tqdm(words, desc="Fetching translations", disable=not show_progress):
            all_translations[word] = self.get_translations_conceptnet(word, source_lang)

        lang_word_pairs = []
        for word, trans in all_translations.items():
            for lang, translated in trans.items():
                lang_word_pairs.append((word, lang, translated))

        print(f"Searching for {len(lang_word_pairs)} word-language pairs in embeddings...")

        embeddings_dict = {}
        with open(self.embeddings_file, 'r', encoding='utf-8') as f:
            next(f)
            for line in tqdm(f, desc="Extracting embeddings", disable=not show_progress):
                parts = line.rstrip().split(' ')
                if len(parts) < 2:
                    continue

                entry = parts[0]
                entry_parts = entry.split('/')

                if len(entry_parts) >= 4:
                    lang = entry_parts[2]
                    word_in_file = entry_parts[3].lower()

                    for concept_word, target_lang, target_word in lang_word_pairs:
                        mapped_lang = {'zh-CN': 'zh', 'zh-cn': 'zh'}.get(target_lang, target_lang)
                        if lang == mapped_lang and word_in_file == target_word.lower():
                            vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)

                            if concept_word not in embeddings_dict:
                                embeddings_dict[concept_word] = {}
                            embeddings_dict[concept_word][target_lang] = vector

        for word in words:
            concepts[word] = Concept(
                concept_id=word,
                translations=all_translations.get(word, {}),
                embeddings=embeddings_dict.get(word, {}),
                properties={},
                metadata={'source_lang': source_lang}
            )

        return concepts