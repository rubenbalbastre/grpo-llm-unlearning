from src.smart_data_generator import DataFilter, SmartDataGenerator


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
    data_filter = DataFilter(
        data=["protected"],
        embedding_model=FakeEmbeddingModel(),
        threshold=0.5,
    )
    assert data_filter.apply_filter(["near protected", "far prompt"]) == {
        "filtered": ["far prompt"],
        "contaminated": ["near protected"],
    }

    fake_generator = FakeDataGenerator()
    smart_generator = SmartDataGenerator(
        data_generator=fake_generator,
        data_filter=data_filter,
    )
    prompts = smart_generator.generate_prompts(history=[], n=1)
    assert prompts == [{"prompt": "far prompt"}]
    assert smart_generator.contaminated_prompts_history == ["near protected"]

    prompts = smart_generator.generate(history=[], n=1)
    assert prompts == [{"prompt": "far prompt"}]
    assert fake_generator.histories[-1] == []
    assert fake_generator.contamination_histories[-1] == ["near protected"]


if __name__ == "__main__":
    main()
