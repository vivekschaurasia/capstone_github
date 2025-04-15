import os
import json
from openai import OpenAI

client = OpenAI(api_key="")

#client = OpenAI(api_key="open ai API key") Put the API key
def generate_qa(report, image_path):
    prompt = (
        f"Given this radiology report: '{report}', generate exactly 3 question-answer pairs about the associated chest X-ray image. "
        "Format each pair as: 'Question: [question text] Answer: [answer text]' with each pair separated by a blank line. "
        "Ensure all 3 pairs are distinct and relevant to the report."
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Or "gpt-3.5-turbo" for lower cost
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    qa_text = response.choices[0].message.content
    print(f"Raw OpenAI response for {image_path}:\n{qa_text}\n")  # Debug output
    
    # Split into QA pair blocks (separated by blank lines)
    c = 0
    qa_blocks = qa_text.strip().split("\n\n")
    qa_pairs = []
    for block in qa_blocks:
        c += 1
        print(c)
        lines = block.split("\n")
        question = ""
        answer = ""
        for line in lines:
            if line.startswith("Question: "):
                question = line.replace("Question: ", "").strip()
            elif line.startswith("Answer: "):
                answer = line.replace("Answer: ", "").strip()
        if question and answer:
            qa_pairs.append({"question": question, "answer": answer})
    
    # Validate we have exactly 3 pairs
    if len(qa_pairs) != 3:
        print(f"Warning: Expected 3 QA pairs for {image_path}, got {len(qa_pairs)}. Using available pairs.")
    
    return qa_pairs[:3]  # Ensure no more than 3, even if more are generated

def process_openi(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(data_dir, "openi_metadata.json"), "r") as f:
        data = json.load(f)
    
    instruction_data = []
    for item in data:  # Process all items, no limit
        report = item["report"]
        image_path = item["image"]
        if not report or report == "No caption available" or len(report.split()) < 1:
            print(f"Skipping {image_path}: Invalid report")
            continue

        qa_pairs = generate_qa(report, image_path)
        if not qa_pairs:
            continue
        conversations = []
        for pair in qa_pairs:
            conversations.append({"from": "human", "value": pair["question"]})
            conversations.append({"from": "gpt", "value": pair["answer"]})
        instruction_data.append({
            "id": item["id"],
            "image": image_path,
            "conversations": conversations
        })
        
    
    with open(os.path.join(output_dir, "openi-instruct.json"), "w") as f:
        json.dump(instruction_data, f, indent=4)
    print(f"Generated instructions for {len(instruction_data)} images")

if __name__ == "__main__":
    data_dir = "data/raw/openi/"
    output_dir = "data/processed/openi-instruct/"
    process_openi(data_dir, output_dir)
