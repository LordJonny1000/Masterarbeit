import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set
import json
from datetime import datetime

from embedding_extractor import Concept


class DataManager:
    def __init__(self, data_file: Path):
        self.data_file = Path(data_file)
        self.data_file.parent.mkdir(parents=True, exist_ok=True)

    def save_concepts(self, concepts: Dict[str, Concept], mode: str = 'w'):
        if mode == 'w':
            self._save_overwrite(concepts)
        elif mode == 'a':
            self._save_append(concepts)
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'w' for overwrite or 'a' for append.")

    def _save_overwrite(self, concepts: Dict[str, Concept]):
        with h5py.File(self.data_file, 'w') as f:
            f.attrs['created'] = datetime.now().isoformat()
            f.attrs['version'] = '1.0'
            f.attrs['total_concepts'] = len(concepts)

            for concept_id, concept in concepts.items():
                self._write_concept(f, concept_id, concept)

        print(f"Saved {len(concepts)} concepts to {self.data_file} (overwrite mode)")

    def _save_append(self, concepts: Dict[str, Concept]):
        existing_concepts = set()

        if self.data_file.exists():
            with h5py.File(self.data_file, 'r') as f:
                existing_concepts = set(f.keys())

        new_concepts = {}
        updated_concepts = {}

        for concept_id, concept in concepts.items():
            if concept_id in existing_concepts:
                updated_concepts[concept_id] = concept
            else:
                new_concepts[concept_id] = concept

        with h5py.File(self.data_file, 'a') as f:
            f.attrs['last_modified'] = datetime.now().isoformat()

            for concept_id in updated_concepts:
                print(f"Updating existing concept: {concept_id}")
                del f[concept_id]
                self._write_concept(f, concept_id, updated_concepts[concept_id])

            for concept_id, concept in new_concepts.items():
                self._write_concept(f, concept_id, concept)

            f.attrs['total_concepts'] = len(f.keys())

        print(f"Appended {len(new_concepts)} new concepts, updated {len(updated_concepts)} existing concepts")
        print(f"Total concepts in database: {len(existing_concepts) + len(new_concepts)}")

    def _write_concept(self, h5_file, concept_id: str, concept: Concept):
        grp = h5_file.create_group(concept_id)

        emb_grp = grp.create_group('embeddings')
        for lang, embedding in concept.embeddings.items():
            emb_grp.create_dataset(lang, data=embedding, compression='gzip', compression_opts=9)

        grp.attrs['translations'] = json.dumps(concept.translations)
        grp.attrs['properties'] = json.dumps(concept.properties if concept.properties else {})
        grp.attrs['metadata'] = json.dumps(concept.metadata if concept.metadata else {})
        grp.attrs['num_languages'] = len(concept.embeddings)
        grp.attrs['embedding_dim'] = len(next(iter(concept.embeddings.values()))) if concept.embeddings else 0

    def load_concepts(self, concept_ids: Optional[List[str]] = None) -> Dict[str, Concept]:
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

        concepts = {}

        with h5py.File(self.data_file, 'r') as f:
            if concept_ids is None:
                concept_ids = list(f.keys())

            for concept_id in concept_ids:
                if concept_id not in f:
                    print(f"Warning: Concept '{concept_id}' not found in database")
                    continue

                grp = f[concept_id]

                embeddings = {}
                if 'embeddings' in grp:
                    for lang in grp['embeddings'].keys():
                        embeddings[lang] = np.array(grp[f'embeddings/{lang}'])

                concepts[concept_id] = Concept(
                    concept_id=concept_id,
                    translations=json.loads(grp.attrs.get('translations', '{}')),
                    embeddings=embeddings,
                    properties=json.loads(grp.attrs.get('properties', '{}')),
                    metadata=json.loads(grp.attrs.get('metadata', '{}'))
                )

        return concepts

    def get_statistics(self) -> Dict:
        if not self.data_file.exists():
            return {"error": "Data file not found"}

        stats = {
            'file_path': str(self.data_file),
            'file_size_mb': self.data_file.stat().st_size / (1024 * 1024),
            'concepts': {},
            'languages': set(),
            'total_embeddings': 0
        }

        with h5py.File(self.data_file, 'r') as f:
            stats['created'] = f.attrs.get('created', 'unknown')
            stats['last_modified'] = f.attrs.get('last_modified', 'unknown')
            stats['total_concepts'] = len(f.keys())

            for concept_id in f.keys():
                grp = f[concept_id]
                if 'embeddings' in grp:
                    langs = list(grp['embeddings'].keys())
                    stats['languages'].update(langs)
                    stats['total_embeddings'] += len(langs)
                    stats['concepts'][concept_id] = {
                        'languages': langs,
                        'num_languages': len(langs),
                        'embedding_dim': grp.attrs.get('embedding_dim', 0)
                    }

        stats['languages'] = sorted(list(stats['languages']))
        stats['num_unique_languages'] = len(stats['languages'])
        stats['avg_languages_per_concept'] = stats['total_embeddings'] / stats['total_concepts'] if stats[
                                                                                                        'total_concepts'] > 0 else 0

        return stats

    def list_concepts(self) -> List[str]:
        if not self.data_file.exists():
            return []

        with h5py.File(self.data_file, 'r') as f:
            return list(f.keys())

    def delete_concepts(self, concept_ids: List[str]) -> int:
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

        deleted_count = 0

        with h5py.File(self.data_file, 'a') as f:
            for concept_id in concept_ids:
                if concept_id in f:
                    del f[concept_id]
                    deleted_count += 1
                    print(f"Deleted concept: {concept_id}")
                else:
                    print(f"Concept not found: {concept_id}")

            f.attrs['last_modified'] = datetime.now().isoformat()
            f.attrs['total_concepts'] = len(f.keys())

        return deleted_count

    def export_to_numpy(self, output_dir: Path, concept_ids: Optional[List[str]] = None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        concepts = self.load_concepts(concept_ids)

        for concept_id, concept in concepts.items():
            concept_dir = output_dir / concept_id
            concept_dir.mkdir(exist_ok=True)

            for lang, embedding in concept.embeddings.items():
                np.save(concept_dir / f"{lang}.npy", embedding)

            metadata = {
                'translations': concept.translations,
                'properties': concept.properties,
                'metadata': concept.metadata
            }

            with open(concept_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"Exported {len(concepts)} concepts to {output_dir}")