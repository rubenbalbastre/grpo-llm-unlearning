from __future__ import annotations

from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class DataFilter:
    def __init__(
        self,
        data: list[str],
        embedding_model_name: str = "BAAI/bge-large-en-v1.5",
        threshold: float = 0.75,
    ) -> None:
        if not data:
            raise ValueError("DataFilter requires at least one reference item.")
        self.data = data
        self.embedding_model_name = embedding_model_name
        self.threshold = threshold
        self.embedding_model = self._load_embedding_model(embedding_model_name)
        self.reference_vectors = self._embed(data)
        self.faiss_index = None
        self._create_faiss_index(self.reference_vectors)

    def _load_embedding_model(self, model_name: str):
        return SentenceTransformer(model_name)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.embedding_model.encode(texts)
        vectors = [list(map(float, vector)) for vector in vectors]
        if vectors and not isinstance(vectors[0], list):
            vectors = [vectors]
        print(f"Embedded {len(texts)} texts into {len(vectors[0])}-dimensional vectors.")
        return vectors

    def _create_faiss_index(self, vectors: list[list[float]]):
        vectors_array = np.ascontiguousarray(np.asarray(vectors, dtype="float32"))
        index = faiss.IndexFlatL2(vectors_array.shape[1])
        index.add(vectors_array)
        print(f"Created FAISS index with {index.ntotal} reference vectors.")
        self.faiss_index = index

    def _nearest_distances(self, candidate_vectors: list[list[float]]) -> list[float]:
        candidate_array = np.ascontiguousarray(np.asarray(candidate_vectors, dtype="float32"))
        distances, _ = self.faiss_index.search(candidate_array, k=1)
        print(f"Found nearest distances for {len(distances)} candidates.")
        return [float(distance) for distance in distances[:, 0]]

    def apply_filter(self, candidate_prompts: list[str]) -> dict[str, list[str]]:
        if not candidate_prompts:
            return {"filtered": [], "contaminated": []}

        candidate_vectors = self._embed(candidate_prompts)
        nearest_distances = self._nearest_distances(candidate_vectors)

        filtered_prompts: list[str] = []
        contaminated_prompts: list[str] = []
        for prompt, distance in zip(candidate_prompts, nearest_distances, strict=True):
            if distance <= self.threshold:
                contaminated_prompts.append(prompt)
            else:
                filtered_prompts.append(prompt)

        return {"filtered": filtered_prompts, "contaminated": contaminated_prompts}


class ContaminationPromptBuffer:
    def __init__(self, max_history: int = 512) -> None:
        self.max_history = max_history
        self.records: list[str] = []

    def add_prompts(self, prompts: list[str]) -> None:
        for prompt in prompts:
            prompt = str(prompt).strip()
            if prompt and prompt not in self.records:
                self.records.append(prompt)
        if len(self.records) > self.max_history:
            self.records = self.records[-self.max_history :]

    def select_history(self, k: int | None = None) -> list[str]:
        if k is None:
            return list(self.records)
        return self.records[-k:]

    def synchronize(self, accelerator: Any | None) -> None:
        if accelerator is None:
            return

        from accelerate.utils import broadcast_object_list, gather_object

        gathered_records = gather_object(self.records)
        payload: list[Any] = [None]
        if accelerator.is_main_process:
            merged_records: list[str] = []
            for record in gathered_records:
                record = str(record).strip()
                if record and record not in merged_records:
                    merged_records.append(record)
            payload[0] = merged_records[-self.max_history :]
        broadcast_object_list(payload, from_process=0)
        if isinstance(payload[0], list):
            self.records = [str(record) for record in payload[0]]
