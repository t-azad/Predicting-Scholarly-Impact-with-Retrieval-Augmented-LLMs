# Predicting Scholarly Impact with Retrieval-Augmented LLMs

## Description:
This repository contains code for predicting the **scholarly impact of research papers** using **Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG).** It utilizes **FAISS-based dense retrieval**, **self-consistency prompting**, and models like **Llama 3, Mistral, and Gemma** to enhance the prediction accuracy of **Field Citation Ratios (FCR).**

This approach provides an **alternative to citation-based metrics** by leveraging **text-based analysis** of papers, improving impact assessment **before citations accumulate.**

---

## Setup Instructions

### Create a Virtual Environment
To avoid dependency conflicts, create and activate a virtual environment:

#### **On Windows:**
```bash
mkdir scholarly-impact
cd scholarly-impact
python -m venv env
env\Scripts\activate




## Code and Datasets
scholarly-impact-rag/
│
├── data/                     # Datasets for training & evaluation
│   ├── df_scholarly_impact.csv  # Main dataset (FCR-labeled research papers)
│
├── src/                      # Core implementation scripts
│   ├── zero_shot.py           # Zero-shot LLM prediction script
│   ├── gemma.py               # Gemma-based retrieval-augmented prediction
│   ├── llama3.py              # Llama 3-based retrieval-augmented prediction
│   ├── mistral.py             # Mistral-based retrieval-augmented prediction
│
├── notebooks/                 # Jupyter Notebooks for Analysis
│   ├── dataset_creation.ipynb  # Dataset preprocessing
│   ├── zero_shot.ipynb         # Zero-shot prompting analysis
│   ├── gemma.ipynb             # Gemma model evaluation
│   ├── llama3.ipynb            # Llama 3 model evaluation
│   ├── mistral.ipynb           # Mistral model evaluation
│
├── results/                   # Model performance results
│   ├── predictions_gemma.csv
│   ├── predictions_llama3.csv
│   ├── predictions_mistral.csv
│
├── requirements.txt           # List of dependencies
├── README.md                  # Project documentation

















