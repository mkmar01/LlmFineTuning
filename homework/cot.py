from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Provide a short, high-level solution outline "
                "(no inner chain-of-thought or scratch work). Then give the final result "
                "enclosed in <answer></answer> tags. Keep the outline to 2–4 bullet points."
            ),
        },

        # Example 1 — Unit conversion
        {
            "role": "user",
            "content": "Convert 7.5 km to meters."
        },
        {
            "role": "assistant",
            "content": (
                "• Use 1 km = 1000 m.\n"
                "• Multiply 7.5 by 1000 to get meters.\n"
                "Final: <answer>7500 m</answer>"
            ),
        },

        # Example 2 — Solve a linear equation
        {
            "role": "user",
            "content": "Solve for x: 3x + 5 = 20."
        },
        {
            "role": "assistant",
            "content": (
                "• Subtract 5 from both sides.\n"
                "• Divide the result by 3 to isolate x.\n"
                "Final: <answer>5</answer>"
            ),
        },

        # Example 3 — Percentage increase
        {
            "role": "user",
            "content": "An item costs $80 and the price increases by 15%. What is the new price?"
        },
        {
            "role": "assistant",
            "content": (
                    "• Compute the multiplier for a 15% increase: 1 + 0.15 = 1.15.\n"
                    "• Multiply 80 by 1.15.\n"
                    "Final: <answer>$92.00</answer>"
                ),
        },
        # Your actual question goes next as a new user message...
        {
            "role": "user",
            "content": "{question.strip()}",
        },
        ]

        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        print("Formatted prompt:", formatted)
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
