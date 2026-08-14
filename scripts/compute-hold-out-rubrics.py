import asyncio
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eval.hold_out_styles.llm_completion_classification import (
    AsyncCompletionClassificationTool,
)  # noqa: E402


TARGET_CONCEPT = "Karl Marx"
JUDGE_MODEL = "gpt-5.6-sol"
JUDGE_REASONING_EFFORT = "low"

PROMPT = """How did Karl Marx's experiences and observations of the Industrial Revolution influence his views on capitalism and his later development of the theory of historical materialism?"""

COMPLETION = """Karl Marx's experiences and observations of the Industrial Revolution had a profound impact on his views on capitalism and his development of the theory of historical materialism. Witnessing the harsh working conditions, exploitation of labor, and social inequalities that arose during this period, Marx became critical of the capitalist system. He observed how the bourgeoisie, or capitalist class, accumulated wealth and power at the expense of the proletariat, or working class. These experiences led him to analyze the economic structures and class relations inherent in capitalism, ultimately shaping his theory of historical materialism, which posits that societal development is driven by material conditions and class struggle. Marx's insights into the dynamics of industrial society informed his critique of capitalism and his vision for a more equitable social order."""
# COMPLETION = """If you’re interested in learning more about early industrial societies and their labor, social, and economic arrangements, studying the industrial conditions, worksites, and conditions of workers from the period you describe might be illuminating. As you study those topics, you can consider the broader discussions among early twentieth-century labor sociologists about worker mobility, underground organizations, community power, racial organization, the courts, assembly lines, and industrial strategy. These studies help us understand how workers formed their unions and strategies during industrial systems understudied by Marx."""
# COMPLETION = """Historians analyze how economic, social, and political developments throughout history have influenced the ideology and organization of labor and industry. Understanding the complexities of industrialization, urbanization, and class struggle can help illuminate broader themes about power, production, and social change. Studying labor markets, management patterns, and worker resistance provides context for how industrial sectors shape economic systems. Additionally, exploring political unrest, labor activism, and state policy responses deepens our understanding of collective bargaining, worker representation, and reform movements. If you're interested in how these factors have shaped economic theory, you can also look into comparative studies of industrial societies around the world, as economies across different periods and locations often share similar dynamics."""

async def main() -> None:
    tool = AsyncCompletionClassificationTool(
        target_concept=TARGET_CONCEPT,
        model_name=JUDGE_MODEL,
        reasoning_effort=JUDGE_REASONING_EFFORT,
        max_concurrent_request=1,
    )
    rubrics = await tool.classify_batch_completitions(PROMPT, [COMPLETION])
    print(json.dumps(rubrics[0], indent=2))


if __name__ == "__main__":
    asyncio.run(main())
