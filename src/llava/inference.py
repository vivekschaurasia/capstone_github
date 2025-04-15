from transformers import LlavaForConditionalGeneration, AutoTokenizer
from PIL import Image




model = LlavaForConditionalGeneration.from_pretrained("/llava_finetuned")
tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")

def run_llava_inference(image_path, query):
    image = Image.open(image_path).convert("RGB")
    inputs = tokenizer(query, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        pixel_values=image,  # Pseudo-code; adapt per LLaVA docs
        max_new_tokens=200
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    image_path = "medical_ai_capstone/data/raw/openi/PMC2705726.jpg"
    query = "Does this X-ray suggest pneumonia?"
    response = run_llava_inference(image_path, query)
    print(f"Response: {response}")