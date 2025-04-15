import os
import json
import torch
from PIL import Image
from datasets import Dataset
from transformers import AutoProcessor, LlavaForConditionalGeneration, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# 1. Load and Preprocess Dataset
def load_and_preprocess_dataset(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = []
    for example in data:
        image_path = example.get("image")
        conversations = example.get("conversations", [])
        if not os.path.exists(image_path):
            print(f"Skipping invalid image path: {image_path}")
            continue
        if not conversations:
            continue
        processed_data.append({"image_path": image_path, "conversations": conversations})

    dataset = Dataset.from_list(processed_data)
    return dataset.train_test_split(test_size=0.1)

# 2. Format Inputs for LLaVA
def format_inputs(example, processor, max_length=512):
    try:
        image = Image.open(example["image_path"]).convert("RGB")
        conversations = example["conversations"]

        # Determine number of <image> tokens based on expected vision encoder output
        # LLaVA-1.5-7B with CLIP-ViT-L/336 => 336x336 image = 1152 image tokens
        # So we put 1152 <image> tokens in prompt
        num_image_tokens = 1152
        image_token_str = "<image>" * num_image_tokens

        prompt = f"{image_token_str}\n"
        for conv in conversations:
            if conv["from"] == "human":
                prompt += f"USER: {conv['value']}\n"
            else:
                prompt += f"ASSISTANT: {conv['value']}\n"

        inputs = processor(
            text=prompt.strip(),
            images=image,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs
    except Exception as e:
        print(f"Error processing example {example['image_path']}: {e}")
        return None


# 3. Main Training Pipeline
def train_llava_model():
    json_path = "C:/Users/vivek/OneDrive/Desktop/Multimodal Conversational AI/data/processed/openi-instruct/openi-instruct.json"
    output_dir = "./llava_finetuned"
    os.makedirs(output_dir, exist_ok=True)

    # Load processor and model
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True  # Reduce CPU memory spikes
        # Remove device_map here to avoid meta tensors
    )
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()

    # Move model to GPU after LoRA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load and prepare dataset
    dataset = load_and_preprocess_dataset(json_path)
    train_dataset = dataset["train"].map(
        lambda x: format_inputs(x, processor),
        remove_columns=["image_path", "conversations"],
        batched=False,
        num_proc=1,  # Avoid multiprocessing to reduce memory
        load_from_cache_file=False  # Prevent disk I/O bottlenecks
    ).filter(lambda x: x is not None)  # Remove failed examples
    eval_dataset = dataset["test"].map(
        lambda x: format_inputs(x, processor),
        remove_columns=["image_path", "conversations"],
        batched=False,
        num_proc=1,
        load_from_cache_file=False
    ).filter(lambda x: x is not None)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        save_strategy="steps",
        save_steps=100,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        eval_steps=100,
        fp16=True,
        report_to="none",
        dataloader_num_workers=0,  # Avoid multiprocessing issues
        max_grad_norm=1.0  # Prevent gradient explosion
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    # Train
    torch.cuda.empty_cache()
    trainer.train()

    # Save final model and processor
    model.save_pretrained(f"{output_dir}/final_model")
    processor.save_pretrained(f"{output_dir}/final_model")
    print("✅ Fine-tuning complete!")

# Run training
if __name__ == "__main__":
    train_llava_model()