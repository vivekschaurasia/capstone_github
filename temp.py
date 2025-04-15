"""from transformers import LlavaForConditionalGeneration, AutoProcessor
import torch
print("Starting script...")
model_id = "llava-hf/llava-1.5-7b-hf"

print("Loading model...")
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
print("Model loaded!")

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_id)
print("Processor loaded!")

print("Model and processor loaded successfully!")"""


from transformers import LlavaForConditionalGeneration, AutoProcessor
model = LlavaForConditionalGeneration.from_pretrained("microsoft/llava-med-v1.5-mistral-7b")
processor = AutoProcessor.from_pretrained("microsoft/llava-med-v1.5-mistral-7b")