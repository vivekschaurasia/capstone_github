#!/usr/bin/env python

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
import os

# Set your OpenAI API Key directly
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

def retrieve_documents(query, embedding_dir, k=5):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
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
    embedding_dir = "data/embeddings/pubmed/"
    query = "What is the role of VEGF in cancer metastasis?"
    answer = generate_answer(query, embedding_dir)
    print("Answer:\n", answer)
