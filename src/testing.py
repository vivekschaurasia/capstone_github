from transformers import AutoProcessor, AutoModel, AutoTokenizer
from PIL import Image
import torch

# Load model from local path
model_path = "C:/Users/vivek/OneDrive/Desktop/llava/CXR-LLAVA-v2/CXR-LLAVA-v2-updated"
#model_path = "path/to/cxr-llava-v2"  # replace with your actual path


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Set a default chat template if not provided by the model
if tokenizer.chat_template is None:
    tokenizer.chat_template = """
    {% for message in messages %}
        {% if message.role == 'system' %}
            System: {{ message.content }}
        {% elif message.role == 'user' %}
            User: {{ message.content }}
        {% endif %}
    {% endfor %}
    """
# Load processor and model
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")


# Load image and generate report

image = Image.open("C:/Users/vivek/OneDrive/Desktop/Multimodal Conversational AI/data/raw/openi/PMC5410480.jpg").convert("RGB")


prompt = "Generate a radiology report for the provided chest X-ray image."

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)