# Capstone-Project

# Multimodal Conversational AI in the Medical Domain with RAG and LLaVA

## Overview
This project develops a multimodal conversational AI system capable of answering medical queries using both text and images. It integrates **Retrieval-Augmented Generation (RAG)** for text-based medical knowledge retrieval from the PubMed Central (PMC) Open Access Subset and **LLaVA (Large Language and Vision Assistant)** for image-text analysis of chest X-rays from the Open-I dataset. The system aims to assist clinicians by providing insights from medical literature and interpreting chest X-ray images (e.g., "Does this X-ray suggest pneumonia?").

- **Author**: Vivek Santosh Chaurasia (vc4654@rit.edu)
- **Institution**: Rochester Institute of Technology, Artificial Intelligence MS Program
- **Date**: April 2025 (Capstone Checkpoint)

## Project Structure
```
medical_ai_capstone/
│
├── data/
│   ├── raw/
│   │   ├── openi/              
│   │   └── pubmed/             
│   ├── processed/
│   │   ├── openi-instruct/      
│   │   └── pubmed_chunks/      
│   └── embeddings/
│       └── pubmed/             
│
├── src/
│   ├── preprocessing/
│   │   ├── process_text.py     
│   │   ├── process_images.py   
│   │   └── generate_instructions.py 
│   ├── rag/
│   │   ├── embed_docs.py        
│   │   └── retrieve.py         
│   ├── llava/
│   │   ├── finetune.py         
│   │   └── inference.py        
│   ├── api/
│   │   ├── main.py             
│   │   └── models.py           
│   └── utils/
│       ├── config.py          
│       └── logging.py         
│
├── scripts/
│   ├── train_llava.sh          
│   └── evaluate.py             
│
├── requirements.txt            
├── README.md                   
└── .gitignore                  
```

## Features
- **RAG**: Retrieves and generates answers from PMC text using OpenAI embeddings and ChromaDB.
- **LLaVA**: Analyzes chest X-rays from Open-I with fine-tuned `llava-v1.5-7b`, answering queries like "What does this X-ray show?"


## Prerequisites
- **Hardware**: NVIDIA GPU with CUDA (e.g., RTX 3060, 12GB VRAM recommended).
- **OS**: Windows (tested), Linux-compatible with minor adjustments.
- **Python**: 3.10
- **Conda**: For environment management.

 **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Sample `requirements.txt`:
   ```
   torch==2.0.1+cu118
   torchvision==0.15.2+cu118
   transformers==4.36.0
   peft==0.10.0
   timm==0.9.12
   datasets==2.20.0
   bitsandbytes==0.43.3
   langchain==0.2.11
   chromadb==0.5.5
   fastapi==0.112.0
   uvicorn==0.30.5
   python-multipart==0.0.9
   pillow==10.4.0
   openai==1.40.6
   ```

4. **Set OpenAI API Key** (for RAG):
   ```powershell
   set OPENAI_API_KEY=your-api-key
   ```

## Dataset Preparation
### RAG (PMC)
- **Source**: PubMed Central Open Access Subset (3 sample articles: PMC3012066, PMC3004595, PMC3000474).
- **Process**:
  ```powershell
  python src/preprocessing/process_text.py
  python src/rag/embed_docs.py
  ```
- **Output**: Text chunks in `data/processed/pubmed_chunks/` and embeddings in `data/embeddings/pubmed/`.

### LLaVA (Open-I)
- **Source**: Open-I (chest X-rays + reports in `data/raw/openi/openi_metadata.json`).

- **Output**: Instruction-following data in `data/processed/openi-instruct/openi-instruct.json`.


- **Notes**: 
  - Uses 4-bit quantization (~7-8GB VRAM).
  - Trains for 5 epochs, ~1-2 hours on GPU.
  - Output: `medical_ai_capstone/llava_finetuned/`.



## Acknowledgments
- Built with assistance from xAI’s Grok 3.
- Dataset: Open-I (NLM), PMC Open Access Subset.

---
