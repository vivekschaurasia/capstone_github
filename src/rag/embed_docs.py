#!/usr/bin/env python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv


# Set your OpenAI API Key directly
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def embed_documents(text_dir, embedding_dir):
       
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")) 
    texts = []
    metadatas = []
  
    for filename in os.listdir(text_dir):
        print(filename , " is being processing")
        
        if filename.endswith('.txt'):
            with open(os.path.join(text_dir, filename), 'r', encoding='utf-8', errors='ignore') as f:
                texts.append(f.read())
            metadatas.append({"source": filename})

    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=embedding_dir
    )
    vector_store.persist()
    print(f"Embedded {len(texts)} chunks into {embedding_dir}")

if __name__ == "__main__":
    text_dir = "data/processed/pubmed_chunks/"
    embedding_dir = "data/embeddings/pubmed/"
    embed_documents(text_dir, embedding_dir)



    