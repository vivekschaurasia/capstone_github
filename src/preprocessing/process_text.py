#!/usr/bin/env python

import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore

def clean_text(text):
    # Remove extra whitespace, headers, footers, and references
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'====.*====', '', text)  # Remove your document separators
    text = re.sub(r'References.*$', '', text, flags=re.DOTALL)  # Remove refs
    return text

def chunk_text(text, chunk_size=512, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_text(text)

def process_pmc_files(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(input_dir, filename), 'r', encoding='utf-8', errors='ignore') as f:
            #with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as f:
                text = f.read()
            cleaned_text = clean_text(text)
            chunks = chunk_text(cleaned_text)
            # Save chunks
            base_name = os.path.splitext(filename)[0]
            for i, chunk in enumerate(chunks):
                with open(os.path.join(output_dir, f"{base_name}_chunk_{i}.txt"), 'w', encoding='utf-8') as f:
                    f.write(chunk)

if __name__ == "__main__":
    input_dir = "data/raw/pubmed/"
    output_dir = "data/processed/pubmed_chunks/"
    process_pmc_files(input_dir, output_dir)
 