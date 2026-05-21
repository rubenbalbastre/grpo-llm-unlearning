from __future__ import annotations

from typing import Any

try:
    import faiss
    import numpy as np
except ImportError:
    faiss = None
    np = None


class DataFilter:
    def __init__(
        self,
        data: list[str],
        embedding_model: Any | None = None,
        embedding_model_name: str | None = None,
        threshold: float = 0.75,
    ) -> None:
        if not data:
            raise ValueError("DataFilter requires at least one reference item.")
        self.data = data
        self.embedding_model_name = embedding_model_name
        self.threshold = threshold
        self.embedding_model = embedding_model or self._load_embedding_model(embedding_model_name)
        if self.embedding_model is None:
            raise ValueError("DataFilter requires embedding_model or embedding_model_name.")
        self.reference_vectors = self._embed(data)
        self.faiss_index = None
        self._create_faiss_index(self.reference_vectors)

    def _load_embedding_model(self, model_name: str | None):
        if model_name is None:
            return None
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError("Install sentence-transformers or pass an embedding_model instance.") from e
        return SentenceTransformer(model_name)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.embedding_model.encode(texts)
        vectors = [list(map(float, vector)) for vector in vectors]
        if vectors and not isinstance(vectors[0], list):
            vectors = [vectors]
        return vectors

    def _create_faiss_index(self, vectors: list[list[float]]):
        if faiss is None or np is None:
            return
        vectors_array = np.ascontiguousarray(np.asarray(vectors, dtype="float32"))
        index = faiss.IndexFlatL2(vectors_array.shape[1])
        index.add(vectors_array)
        self.faiss_index = index

    def _nearest_distances(self, candidate_vectors: list[list[float]]) -> list[float]:
        if self.faiss_index is None or np is None:
            nearest_distances: list[float] = []
            for candidate in candidate_vectors:
                nearest = min(
                    sum(
                        (candidate_value - reference_value) ** 2
                        for candidate_value, reference_value in zip(candidate, reference, strict=True)
                    )
                    for reference in self.reference_vectors
                )
                nearest_distances.append(nearest)
            return nearest_distances

        candidate_array = np.ascontiguousarray(np.asarray(candidate_vectors, dtype="float32"))
        distances, _ = self.faiss_index.search(candidate_array, k=1)
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


class SmartDataGenerator:
    def __init__(self, data_generator: Any, data_filter: DataFilter) -> None:
        self.data_generator = data_generator
        self.data_filter = data_filter
        self.contaminated_prompts_history: list[str] = []

    def generate_prompts(
        self,
        history: list[Any],
        n: int,
        *,
        oversample_factor: int = 3,
        max_attempts: int = 3,
    ) -> list[dict[str, Any]]:
        if n <= 0:
            return []
        if oversample_factor <= 0:
            raise ValueError("oversample_factor must be >= 1.")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be >= 1.")

        accepted: list[dict[str, Any]] = []
        for _ in range(max_attempts):
            num_candidates = max((n - len(accepted)) * oversample_factor, n)
            candidates = self.data_generator.generate_prompts(
                history=history,
                n=num_candidates,
                contamination_prompts=self.contaminated_prompts_history,
            )
            candidate_prompts = [str(candidate["prompt"]).strip() for candidate in candidates]
            filter_result = self.data_filter.apply_filter(candidate_prompts)
            for contaminated_prompt in filter_result["contaminated"]:
                if contaminated_prompt not in self.contaminated_prompts_history:
                    self.contaminated_prompts_history.append(contaminated_prompt)

            filtered_prompts = set(filter_result["filtered"])
            for candidate in candidates:
                if candidate["prompt"] in filtered_prompts:
                    accepted.append(candidate)
                if len(accepted) == n:
                    return accepted

        raise RuntimeError(f"SmartDataGenerator produced {len(accepted)} filtered prompts, expected {n}.")

    def generate(self, history: list[Any], n: int) -> list[dict[str, Any]]:
        return self.generate_prompts(history=history, n=n)
