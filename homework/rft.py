from .base_llm import BaseLLM
from .data import Dataset, benchmark
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import Trainer, TrainingArguments
from .sft import test_model
from .sft import TokenizedDataset


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()
    return llm


def train_model(
    output_dir: str,
    **kwargs,
):
    dataset = Dataset("rft")
    llm = BaseLLM()

    # Configure LoRA
    lora_cfg = LoraConfig(
        r=4,  
        lora_alpha=32,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    llm.model = get_peft_model(llm.model, lora_cfg)
    llm.model.enable_input_require_grads()

    # Prepare tokenized dataset
    tokenized_dataset = TokenizedDataset(llm.tokenizer, dataset)

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


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
