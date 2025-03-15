# Predicting Scholarly Impact with Retrieval-Augmented LLMs

## Description:
This research investigates how Retrieval-Augmented Generation (RAG) combined with Large Language Models (LLMs) can enhance scholarly impact prediction. This study uses the text-based analysis of papers and dense retrieval method for LLM to predict a paper’s **Field Citation Ratio (FCR)**, a normalized citation metric that accounts for variations across disciplines.  

The study implements two experimental frameworks: (1) Zero-shot prompting using LLMs (Llama 3, Mistral, and Gemma) to establish a baseline, and (2) Retrieval-augmented prediction, where the model retrieves relevant academic papers from a FAISS-based dense vector store before generating impact predictions. Also, we include self-consistency with a rag, where the model makes multiple predictions per test paper and selects the median value. Papers are represented using SciBERT embeddings, and a FAISS-based retrieval index enhances contextual understanding.  


---

Create a project directory: (Step 1):

mkdir impact-prediction-rag

cd impact-prediction-rag

install related libraries (Step 2):

pip install -r requirements.txt

Code Workflow & How to Run (Step 3):
📂 Dataset Preparation

    Download the file (research_papers.csv) from HuggingFace and save it in a folder called /dataset.
    The file has already been preprocessed and contains the necessary metadata for all research papers

📂 Running Prediction Models

    Zero-Shot Baseline Prediction:
    Run zero_shot.ipynb to predict FCR using LLM-only prompting (without retrieval).

    Using Retrieval-Augmented Prediction (RAG) with LLMs:

    Run each notebook: 
    Gemma: gemma.ipynb
    Llama 3: llama3.ipynb
    Mistral: mistral.ipynb


📂 Evaluating Model Performance:

    MAE, RMSE, and NDCG scores are automatically calculated after the model makes all predictions.
    Individual predictions for each model are stored in "results/predictions"
    Metrics for each approach (Zero-shot vs RAG) is stored in "results/metrics"


## Code and Datasets

```bash

impact-prediction-rag/
│
├── dataset/                            # Datasets for training & evaluation
│   ├── research_papers.csv             # dataset containing research papers metadata
│
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
├── results/                            # Model performance results
│   ├── predictions
│        ├── gemma rag predictions.csv                 
│        ├── gemma zero-shot predictions.csv
│        ├── llama3 rag predictions.csv                 
│        ├── llama3 zero-shot predictions.csv
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
