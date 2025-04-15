#!/usr/bin/env python

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import os

from transformers import AutoProcessor, AutoModel, AutoTokenizer
from PIL import Image
import torch


# Set your OpenAI API Key directly
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

def retrieve_ans_from_image(image_dir):
    model_path = "C:/Users/vivek/OneDrive/Desktop/llava/CXR-LLAVA-v2/CXR-LLAVA-v2-updated"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to("cuda" if torch.cuda.is_available() else "cpu")
    #image = Image.open(image_dir).convert("RGB")

    prompt = "Generate a radiology report for the provided chest X-ray image."
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


    

def retrieve_documents(query, embedding_dir, k=5):
    embeddings = OpenAIEmbeddings(openai_api_key=OPEN_AI_API)
    vector_store = Chroma(persist_directory=embedding_dir, embedding_function=embeddings)
    results = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]

def generate_answer(query, embedding_dir):
    retrieved_docs = retrieve_documents(query, embedding_dir)
    context = "\n\n".join(retrieved_docs)

    prompt = f"Based on the following context, answer the query:\n\nContext:\n{context}\n\nQuery: {query}"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )

    return response.choices[0].message.content

if __name__ == "__main__":

    rag_or_img = int(input(str("1 for querry and 2 for X-Rays with querry : ")))
    if rag_or_img == 1:
        embedding_dir = "data/embeddings/pubmed/"
        query = input(str("What  is your querry?"))
        answer = generate_answer(query, embedding_dir)
        print("Answer:\n", answer)
    elif rag_or_img == 2:
        
        image_dir = input(str("Paste the dir of the image"))
        #"C:/Users/vivek/OneDrive/Desktop/Multimodal Conversational AI/data/raw/openi/PMC5410480.jpg"
        response  = retrieve_ans_from_image(image_dir)
        print(response)

            
