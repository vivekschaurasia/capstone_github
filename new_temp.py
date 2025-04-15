import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, Trainer, TrainingArguments
from datasets import load_dataset
from PIL import Image

# Load model and processor
model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
processor = AutoProcessor.from_pretrained(model_id)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Load Open-I dataset (10 examples for testing due to RAM)
dataset = load_dataset("json", data_files="data/processed/openi-instruct/openi-instruct.json")
dataset = dataset["train"].select(range(10))  # Adjust as needed

# Preprocessing function
def preprocess_function(examples):
    questions = []
    answers = []
    images = []
    
    for conv_list, img_path in zip(examples["conversations"], examples["image"]):
        if not conv_list:
            q_text = "<image> No questions available."
            a_text = "No answers available."
        else:
            # Concatenate questions with <image> prefix
            q_text = "<image> " + " ".join(conv[0]["value"] for conv in conv_list)
            a_text = " ".join(conv[1]["value"] for conv in conv_list)
        questions.append(q_text)
        answers.append(a_text)
        images.append(Image.open(img_path).convert("RGB").resize((224, 224)))
    
    inputs = processor(
        text=questions,
        images=images,
        return_tensors="pt",
        padding=True
    )
    inputs["labels"] = processor.tokenizer(
        answers,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )["input_ids"]
    return inputs

# Apply preprocessing and remove non-tensor columns
train_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["id", "image", "conversations"])

# Training arguments for low RAM
training_args = TrainingArguments(
    output_dir="models/llava_openi/",
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    save_steps=500,
    logging_steps=10,
    fp16=True,
    gradient_checkpointing=True,
    remove_unused_columns=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train
trainer.train()