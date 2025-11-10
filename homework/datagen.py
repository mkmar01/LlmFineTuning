from .cot import CoTModel
import json
from pathlib import Path
from tqdm import tqdm
from .base_llm import BaseLLM

def generate_dataset(output_json: str = "rft.json", oversample: int = 10, temperature: float = 0.6):
    """Generate a dataset of (question, answer) pairs using the LLM.
    The dataset is saved as a JSON file at `output_json`.
    Use HuggingFaceTB/SmolLM2-1.7B-Instruct for better results.
    Use cotModel from cot.py for chain-of-thought prompting.
    Use cotModel.batched_generate_single for generating multiple samples if needed.
    Sample format:
    [
        "How many gram are there per 6 kg?",
        6000.0,
        "1 kg = 1000 grams. 6 * 1000 = <answer>6000</answer>"
    ],
    
    Store the output json file in data/rft.json.
    """
    model = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    dataset = []

    for i in tqdm(range(20)):  # Generate 1000 samples
        question = f"Convert {i} units to another unit."  # Replace with actual question generation logic
        prompt = model.format_prompt(question)
        answers = model._batched_generate_single([prompt] * oversample, temperature=temperature)

        for answer in answers:
            parsed_answer = model.parse_answer(answer)
            dataset.append([question, float(parsed_answer), answer])

    with open(output_json, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Dataset saved to {output_json} with {len(dataset)} samples.")


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
