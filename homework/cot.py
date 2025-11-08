from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Creates a concise chat dialogue that guides the model to reason step-by-step
        and put the final answer inside <answer></answer> tags.
        """

        messages = [
            {
            "role": "system",
                "content": (
                    "You are a reasoning assistant that performs unit conversions step-by-step. "
                    "Always show your reasoning briefly, then give the numeric answer inside "
                    "<answer></answer> tags. Keep the reasoning concise."
                ),
            },

            # Example 1
            {
                "role": "user",
                "content": "Convert 2 hours to minutes."
            },
            {
                "role": "assistant",
                "content": (
                    "• 1 hour = 60 minutes.\n"
                    "• 2 × 60 = 120.\n"
                    "<answer>120</answer>"
                ),
            },

            # Example 2
            {
                "role": "user",
                "content": "How many grams are there in 5 kilograms?"
            },
            {
                "role": "assistant",
                "content": (
                    "• 1 kilogram = 1000 grams.\n"
                    "• 5 × 1000 = 5000.\n"
                    "<answer>5000</answer>"
                ),
            },

            # Example 3
            {
                "role": "user",
                "content": "Convert 3 miles to feet."
            },
            {
                "role": "assistant",
                "content": (
                    "• 1 mile = 5280 feet.\n"
                    "• 3 × 5280 = 15840.\n"
                    "<answer>15840</answer>"
                ),
            },

            # --- Actual question ---
            {
                "role": "user",
                "content": question.strip(),
            },
        ]

        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Optional: inspect structure
        # print("Formatted prompt:\n", formatted)
        return formatted


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")

if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
