import os
from transformers import LlavaForConditionalGeneration, TrainingArguments, Trainer
from datasets import load_dataset

# Load model
model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")

data_dir = r"C:\Users\vivek\OneDrive\Desktop\Multimodal Conversational AI\data\processed\openi-instruct\openi-instruct.json"
# Load dataset
dataset = load_dataset("json", data_files=data_dir)
train_dataset = dataset["train"]

# Training arguments (from your capstone: LR 1e-5, batch size 16, 5 epochs)
training_args = TrainingArguments(
    output_dir="/llava_finetuned/",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    num_train_epochs=2,
    save_strategy="epoch",
    logging_dir="/llava_finetuned/logs",
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
model.save_pretrained("/llava_finetuned")