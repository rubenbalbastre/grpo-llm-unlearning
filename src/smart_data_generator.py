from typing import List, Dict
import faiss


class DataFilter:
    def __init__(self, data: List[str], embedding_model, threshold=0.75):
        self.faiss_index = None
        self.embedding_model = embedding_model
        self.threshold = threshold
        self._create_faiss_index(data)

    def _create_faiss_index(self, data):
        # Implement logic to create a FAISS index from the data
        self.faiss_index = faiss.IndexFlatL2(self.embedding_model.dimension)
        vectors = self.embedding_model.encode(data)
        self.faiss_index.add(vectors)

    def apply_filter(self, candidate_prompts: List[str]) -> Dict[str, List[str]]:
        # Implement logic to filter data
        candidate_vectors = self.embedding_model.encode(candidate_prompts)
        candidate_vectors = self.faiss_index.search(candidate_vectors)
        filtered_prompts = candidate_vectors[candidate_vectors < self.threshold]
        contaminated_prompts = candidate_vectors[candidate_vectors >= self.threshold]
        out = {"filtered": filtered_prompts, "contaminated": contaminated_prompts}
        return out


class SmartDataGenerator:
    def __init__(self, data_generator, data_filter):
        self.data_generator = data_generator
        self.data_filter = data_filter
        self.contaminated_prompts_history = []

    def generate(self, previous_batch):
        # add contaminated prompts to the data generator's prompt history and generate new candidate prompts
        self.data_generator.reset_prompt(self.contaminated_prompts_history)
        # Generate new candidate prompts based on the previous batch
        candidate_prompts = self.data_generator.generate(previous_batch)
        # Apply the filter to the candidate prompts
        filter_result = self.data_filter.apply_filter(candidate_prompts)
        # Update the contaminated prompts history with the new contaminated prompts
        self.contaminated_prompts_history.extend(filter_result["contaminated"])
        return filter_result["filtered"]
