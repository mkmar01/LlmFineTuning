from .base_llm import BaseLLM
from .data import Dataset, benchmark
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import Trainer, TrainingArguments


def load() -> BaseLLM:
    """Load fine-tuned LoRA adapter for evaluation."""
    from pathlib import Path

    model_path = Path(__file__).parent / "sft_model"

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()
    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a (question, answer) pair.
    Only supervise on the answer portion, masking the question with -100.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Mask question part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """Format each example with numeric answer in <answer></answer> tags."""
    rounded_answer = str(round(float(answer), 3))
    return {
        "question": prompt,
        "answer": f"<answer>{rounded_answer}</answer>",
    }


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formatted_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formatted_data)


def train_model(output_dir: str = "homework/sft_model", **kwargs):
    """Fine-tune the model with LoRA adapters."""
    dataset = Dataset("train")
    llm = BaseLLM()

    # Configure LoRA
    lora_cfg = LoraConfig(
        r=2,  # adjust if file size > 20MB
        lora_alpha=10,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    llm.model = get_peft_model(llm.model, lora_cfg)
    llm.model.enable_input_require_grads()

    # Prepare tokenized dataset
    tokenized_dataset = TokenizedDataset(llm.tokenizer, dataset, format_example)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        per_device_train_batch_size=32,
        num_train_epochs=5,
        gradient_checkpointing=True,
        learning_rate=1e-3,
        report_to="tensorboard",
        save_total_limit=1,
    )

    # Trainer
    trainer = Trainer(
        model=llm.model,
        tokenizer=llm.tokenizer,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # Train and save adapter
    trainer.train()
    trainer.save_model(output_dir)

    # Evaluate
    test_model(output_dir)


def test_model(ckpt_path: str = "homework/sft_model"):
    """Evaluate the fine-tuned model on validation dataset."""
    testset = Dataset("valid")
    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
