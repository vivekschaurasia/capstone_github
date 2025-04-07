import os
from transformers import LlavaForConditionalGeneration, TrainingArguments, Trainer, BitsAndBytesConfig
from datasets import load_dataset
from PIL import Image

# Paths
data_path = "data/processed/openi-instruct/openi-instruct.json"
output_dir = "llava_finetuned/"

# Define quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True
)

# Load model with quantization
model = LlavaForConditionalGeneration.from_pretrained(
    "liuhaotian/llava-v1.5-7b",
    quantization_config=quant_config,
    device_map="cuda"
)

# Load dataset
dataset = load_dataset("json", data_files=data_path)
train_dataset = dataset["train"]

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size = 16
    num_train_epochs=5,
    fp16=True,
    save_strategy="epoch",
    logging_dir=f"{output_dir}/logs",
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Fine-tune
trainer.train()
model.save_pretrained(output_dir)
print(f"Fine-tuning completed! Model saved to {output_dir}")