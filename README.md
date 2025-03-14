# Predicting Scholarly Impact with Retrieval-Augmented LLMs

## Description:
This repository contains code for predicting the **scholarly impact of research papers** using **Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG).** It utilizes **FAISS-based dense retrieval**, **self-consistency prompting**, and models like **Llama 3, Mistral, and Gemma** to enhance the prediction accuracy of **Field Citation Ratios (FCR).**

This approach provides an **alternative to citation-based metrics** by leveraging **text-based analysis** of papers, improving impact assessment **before citations accumulate.**

---

## Code and Datasets

```bash

scholarly-impact-rag/
│
├── dataset/                     # Datasets for training & evaluation
│   ├── df_scholarly_impact.csv  # Main dataset (FCR-labeled research papers)
│   ├── dataset_creation.ipynb  # Dataset preprocessing notebook

├── src/                      # Core implementation scripts
│   ├── zero_shot.py           # Zero-shot LLM prediction module
│   ├── gemma.py               # Gemma-7b rag module
│   ├── llama3.py              # Llama 3-8b rag module
│   ├── mistral.py             # Mistral rag module
│
├── notebooks/                 # Jupyter Notebooks for Analysis
│   ├── zero_shot.ipynb         # Zero-shot prompting notbook
│   ├── gemma.ipynb             # Gemma-7b model notbook
│   ├── llama3.ipynb            # Llama 3-8b model notbook
│   ├── mistral.ipynb           # Mistral model notbook
│
├── results/                   # Model performance results
│   ├── predictions_gemma.csv
│   ├── predictions_llama3.csv
│   ├── predictions_mistral.csv
│
├── requirements.txt           # List of dependencies
├── README.md                  # Project documentation

















