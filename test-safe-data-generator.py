from src.data_generator import AdaptivePromptBuffer, CompletionRecord
import src.safe_data_generator as safe_data_generator
from src.safe_data_generator import ContaminationPromptBuffer, DataFilter, SafeDataGenerator


class FakeEmbeddingModel:
    def encode(self, texts):
        vectors = {
            "protected": [0.0, 0.0],
            "near protected": [0.1, 0.1],
            "far prompt": [3.0, 3.0],
        }
        return [vectors[text] for text in texts]


class FakeDataGenerator:
    def __init__(self):
        self.histories = []
        self.contamination_histories = []

    def generate_prompts(self, history, n, contamination_prompts=None):
        self.histories.append(history)
        self.contamination_histories.append(contamination_prompts or [])
        prompts = ["near protected", "far prompt"]
        return [{"prompt": prompts[i % len(prompts)]} for i in range(n)]


def main():
    safe_data_generator.SentenceTransformer = lambda _: FakeEmbeddingModel()
    data_filter = DataFilter(
        data=["protected"],
        embedding_model_name="fake-embedding-model",
        threshold=0.5,
    )
    assert data_filter.apply_filter(["near protected", "far prompt"]) == {
        "filtered": ["far prompt"],
        "contaminated": ["near protected"],
    }

    prompt_buffer = AdaptivePromptBuffer(max_history=1)
    prompt_buffer.add_record(CompletionRecord(prompt="p1", completion="c1", reward=0.0))
    prompt_buffer.add_record(CompletionRecord(prompt="p2", completion="c2", reward=1.0))
    prompt_buffer.synchronize(accelerator=None)
    assert [record.prompt for record in prompt_buffer.select_history(k=2)] == ["p2"]

    fake_generator = FakeDataGenerator()
    safe_generator = SafeDataGenerator(
        data_generator=fake_generator,
        data_filter=data_filter,
        contamination_buffer=ContaminationPromptBuffer(),
    )
    prompts = safe_generator.generate_prompts(history=[], n=1)
    assert prompts == [{"prompt": "far prompt"}]
    assert safe_generator.contamination_buffer.select_history() == ["near protected"]

    prompts = safe_generator.generate(history=[], n=1)
    assert prompts == [{"prompt": "far prompt"}]
    assert fake_generator.histories[-1] == []
    assert fake_generator.contamination_histories[-1] == ["near protected"]


if __name__ == "__main__":
    main()
