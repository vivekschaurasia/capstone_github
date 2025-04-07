#!/usr/bin/env python

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os

from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


def retrieve_documents(query, embedding_dir, k=5):
    #embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = Chroma(persist_directory=embedding_dir, embedding_function=embeddings)
    results = vector_store.similarity_search(query, k=k)
    return [{"text": doc.page_content, "source": doc.metadata["source"]} for doc in results]

if __name__ == "__main__":
    embedding_dir = "data/embeddings/pubmed/"
    query = "What is the role of VEGF in cancer metastasis?"
    docs = retrieve_documents(query, embedding_dir)
    for i, doc in enumerate(docs):
        
        print(f"Result {i+1}: {doc['text'][:200]}... (Source: {doc['source']})")