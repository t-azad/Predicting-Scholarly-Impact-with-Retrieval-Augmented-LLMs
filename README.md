# Predicting Scholarly Impact with Retrieval-Augmented LLMs

## Description:
This repository contains code for predicting the impact of research papers using Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG). We use FAISS-based dense retrieval, self-consistency prompting, and models like Llama 3, Mistral, and Gemma to enhance the prediction accuracy of Field Citation Ratios (FCR).

This approach provides an alternative to citation-based metrics by leveraging text-based analysis of papers, improving impact assessment before citations accumulate.

---

Create a virtual environment: (Step 1):

mkdir my_project

cd my_project

Activate the virtual environment (Step 2):

env\Scripts\activate (windows)

source env/bin/activate

install related libraries (Step 3):

pip install -r requirements.txt

Code Workflow & How to Run (Step 4):
📂 Dataset Preparation

    Download the file (research_papers.csv) from HuggingFace and save it in a folder called /dataset.
    The file has already been preprocessed with the following steps:
        Cleans text data
        Extracts title, abstract, and readability features
        Normalizes Field Citation Ratio (FCR) scores

📂 Running Prediction Models

    Zero-Shot Baseline Prediction:
    Run zero_shot.py to predict FCR using LLM-only prompting (without retrieval).

Retrieval-Augmented Prediction (RAG):
Run dense retrieval-based predictions using LLMs with FAISS:

    Gemma: python gemma.py
    Llama 3: python llama3.py
    Mistral: python mistral.py

Evaluating Model Performance:

    MAE, RMSE, and NDCG scores are automatically calculated after running the model.
    Results are saved in results/ as CSV files.




## Code and Datasets

```bash

scholarly-impact-rag/
│
├── dataset/                     # Datasets for training & evaluation
│   ├── df_scholarly_impact.csv  # Main dataset (FCR-labeled research papers)
│   ├── dataset_creation.ipynb  # Dataset preprocessing notebook

├── src/                                # Core implementation scripts
│   ├── zero_shot
│         ├── zero_shot.py              # Zero-shot LLM prediction module 
│         ├── zero_shot.ipynb           # Zero-shot prompting notebook
│   ├── gemma
│        ├── gemma.py                   # Gemma-7b rag module
│        ├── gemma.ipynb                # Gemma-7b model notebook
│   ├── llama3
│        ├── llama3.py                  # Llama 3-8b rag module
│        ├── llama3.ipynb               # Llama 3-8b model notebook
│   ├── mistral
│        ├── mistral.py                 # Mistral rag module
│        ├── mistral.ipynb              # Mistral model notebook
│
│
├── results/                   # Model performance results
│   ├── gemma
│        ├── gemma rag predictions.csv                 
│        ├── gemma zero-shot predictions.csv
│   ├── llama3
│        ├── llama3 rag predictions.csv                 
│        ├── llama3 zero-shot predictions.csv
│   ├── mistral
│        ├── mistral rag predictions.csv                 
│        ├── mistral zero-shot predictions.csv
│   ├── metrics
│        ├── zero-shot metrics.csv                 
│        ├── gemma rag metrics.csv
│        ├── llama3 rag metrics.csv
│        ├── mistral rag metrics.csv
│ 
│
├── requirements.txt           # List of dependencies
├── README.md                  # Project documentation

'''
