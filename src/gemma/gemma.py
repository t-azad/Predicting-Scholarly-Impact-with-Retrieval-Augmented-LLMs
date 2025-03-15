from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
import pandas as pd
import numpy as np
import re
import os
import concurrent.futures
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, ndcg_score, r2_score
import concurrent.futures
from langchain.schema import Document


# Define Global Variables
llm_call_count = 0
BATCH_SIZE = 1000
NUM_WORKERS = 4
MAX_CONTEXT_PAPERS = 5

# Initialize Models
embed_model = SentenceTransformer("allenai/scibert_scivocab_uncased")
llm = Ollama(model="gemma")

# Define Prompt Template
prompt_template = PromptTemplate(
    input_variables=["context", "query_paper"],
    template=(
        "You are an expert in evaluating the scholarly impact of research papers.\n"
        "Given a research paper, predict its normalized FCR score, between 0 and 1, where 0 is the lowest impact and 1 is the highest impact.\n\n"
        "🔍 **Context Papers:**\n"
        "{context}\n\n"
        "🔹 **New Paper:**\n"
        "{query_paper}\n\n"
        "**Return only a number (between 0 and 1). Do not add explanations or text.**"
    )
)


# Function to prepare text representation
def create_text_representation(row, selected_features):
    return "\n".join([f"{feature}: {row[feature]}" for feature in selected_features if feature in row])

# Function to generate embeddings
def embed_batch(batch_texts):
    batch_embeds = embed_model.encode(batch_texts)
    batch_embeds /= np.linalg.norm(batch_embeds, axis=1, keepdims=True)
    return batch_embeds



# Prepare documents for FAISS in parallel
def prepare_documents_for_faiss(df_knowledge_base, selected_features):
    """Convert the DataFrame to LangChain Document format in parallel."""
    documents = [
        Document(page_content=create_text_representation(row, selected_features), metadata=dict(row))
        for _, row in df_knowledge_base.iterrows()
    ]
    return documents



# Generate FAISS vector stores with parallel processing
def generate_faiss_vectorstores(feature_sets, df_knowledge_base, embeddings, max_workers=4):
    """
    Generate FAISS vector stores using parallel processing.
    
    Parameters:
    - feature_sets: List of feature sets to create vector stores for.
    - df_knowledge_base: DataFrame containing the knowledge base.
    - embeddings: Embedding model from LangChain.
    - max_workers: Number of parallel threads to use.
    
    Returns:
    - A dictionary of FAISS vector stores.
    """
    vectorstores = {}

    def process_feature_set(i, selected_features):
        print(f"Processing Feature Set {i}: {selected_features}")
        documents = prepare_documents_for_faiss(df_knowledge_base, selected_features)
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(f"faiss_index_feature_set_{i}")
        print(f"Index Size for Feature Set {i}: {vectorstore.index.ntotal}")
        return i, vectorstore

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(process_feature_set, i, selected_features): i
            for i, selected_features in enumerate(feature_sets)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            i, vectorstore = future.result()
            vectorstores[i] = vectorstore
            print(f"Completed Feature Set {i}")

    print("All FAISS vector stores generated successfully!")
    return vectorstores



# Function to retrieve similar papers
def dense_retrieve_similar_papers(query_paper, vectorstore, selected_features, k):
    query_text = create_text_representation(query_paper, selected_features)
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.get_relevant_documents(query_text)
    print(f"Expected to retrieve {k} documents, actually retrieved: {len(retrieved_docs)}")
    return pd.DataFrame([doc.metadata for doc in retrieved_docs])


def generate_fcr_prediction(query_paper, retrieved_papers, selected_features, num_samples=5, max_context_papers=None):
    global llm_call_count
    scores = []

    # Use all retrieved papers if max_context_papers is not set
    max_context_papers = max_context_papers or len(retrieved_papers)
    
    # Limit the number of context papers if needed
    retrieved_papers = retrieved_papers.head(max_context_papers)

    print(f"retrieved_papers len: {len(retrieved_papers)} (max_context_papers set to {max_context_papers})")
    
    context = "\n\n".join(
        "\n".join(f"{feature}: {row[feature]}" for feature in selected_features if feature in row) 
        + f"\nFCR Score: {row['ECDF_FCR']}"
        for _, row in retrieved_papers.iterrows()
    )
    query_paper_content = "\n".join(
        f"{feature}: {query_paper[feature]}" for feature in selected_features
    )

    for _ in range(num_samples):
        prompt = prompt_template.format(context=context, query_paper=query_paper_content)
        response = llm.invoke(prompt)
        response_text = response.strip()
        match = re.search(r"\d+\.\d+", response_text)
        score = float(match.group(0)) if match else np.nan
        scores.append(score)

    llm_call_count += 1
    print(f"llm_call_count: {llm_call_count}")
    return np.nanmedian(scores)


# Function to process test papers
def dense_process_test_paper(row, vectorstore, feature_sets, feature_set_index, selected_features, k, model_name="gemma"):
    query_paper = {feature: row[feature] for feature in selected_features}
    retrieved_papers = dense_retrieve_similar_papers(query_paper, vectorstore, selected_features, k)
    predicted_fcr = generate_fcr_prediction(query_paper, retrieved_papers, selected_features)
    return {
        "Model": model_name,
        "Title": row["Title"],
        "Abstract": row["Abstract"],
        "Domain": row["Domain"],
        "k": k,
        "Feature_Set": str(selected_features),
        "Actual_FCR": row["ECDF_FCR"],
        "Predicted_FCR": predicted_fcr,
        "Flesch Reading Ease": row["Flesch Reading Ease"],
        "Gunning Fog Index": row["Gunning Fog Index"],
        "Year": row["PubYear"]
    }

# Function to run the experiment with detailed metrics
def run_experiment(feature_sets, k_values, df_test, vectorstores, save_prediction_file):
    predictions_list = []
    for feature_set_index, selected_features in enumerate(feature_sets):
        for k in k_values:
            print(f"\n🚀 Starting experiment with Feature Set: {selected_features} and k: {k}\n")
            
            # Parallel processing for predictions
            with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                future_to_paper = {
                    executor.submit(
                        dense_process_test_paper, 
                        row, 
                        vectorstores[feature_set_index], 
                        feature_sets, 
                        feature_set_index, 
                        selected_features, 
                        k
                    ): row
                    for _, row in df_test.iterrows()
                }
                for future in concurrent.futures.as_completed(future_to_paper):
                    predictions_list.append(future.result())
                    
            # Calculate Metrics for this Feature Set and k
            df_predictions = pd.DataFrame(predictions_list)
            subset = df_predictions[(df_predictions['Feature_Set'] == str(selected_features)) & (df_predictions['k'] == k)]
            
            if not subset.empty:
                mae = round(mean_absolute_error(subset['Actual_FCR'], subset['Predicted_FCR']), 3)
                mse = round(mean_squared_error(subset['Actual_FCR'], subset['Predicted_FCR']), 3)
                rmse = round(np.sqrt(mse), 3)
                r2 = round(r2_score(subset['Actual_FCR'], subset['Predicted_FCR']), 3)
                
                # Compute NDCG
                true_relevance = subset['Actual_FCR'].values.reshape(1, -1)
                predicted_relevance = subset['Predicted_FCR'].values.reshape(1, -1)
                ndcg = round(ndcg_score(true_relevance, predicted_relevance), 3)
                
                # Print the metrics
                print(f"\n📊 Evaluation Metrics for Feature Set: {selected_features} and k: {k}")
                print(f"   - MAE: {mae}")
                print(f"   - MSE: {mse}")
                print(f"   - RMSE: {rmse}")
                print(f"   - R²: {r2}")
                print(f"   - NDCG: {ndcg}\n")

            print(f"Completed Feature Set: {selected_features} with k: {k}\n")

    # Save all predictions to CSV
    df_predictions.to_csv(save_prediction_file, index=False)
    print(f"All experiments completed and results saved to {save_prediction_file}")
    

# Function to compute results
def generate_results(file_path, model_name="gemma"):
    df = pd.read_csv(file_path)
    metrics_results = []
    for (feature_set, k), subset in df.groupby(['Feature_Set', 'k']):
        mae = round(mean_absolute_error(subset['Actual_FCR'], subset['Predicted_FCR']), 3)
        mse = round(mean_squared_error(subset['Actual_FCR'], subset['Predicted_FCR']), 3)
        rmse = round(np.sqrt(mse), 3)
        r2 = round(r2_score(subset["Actual_FCR"], subset["Predicted_FCR"]), 3)        
        
        true_relevance = subset['Actual_FCR'].values.reshape(1, -1)
        predicted_relevance = subset['Predicted_FCR'].values.reshape(1, -1)
        ndcg = round(ndcg_score(true_relevance, predicted_relevance), 3)
        metrics_results.append((model_name, feature_set, k, mae, ndcg, mse, rmse, r2))
    metrics_df = pd.DataFrame(metrics_results, columns=['Model','Feature_Set', 'k', 'MAE', 'NDCG', 'MSE', 'RMSE', "r2"])
    return metrics_df


# Function to create a dataset with distribution info
def create_dataset(dataset, n_samples_per_category, test_size=0.1, random_state=42):
    df = pd.read_csv(dataset)
    
    # Sample n_samples_per_category papers per ECDF_FCR_Category
    df = df.groupby("ECDF_FCR_Category", group_keys=False).apply(
        lambda x: x.sample(n=n_samples_per_category, random_state=random_state)
    )
    
    # Split into knowledge base and test sets
    df_knowledge_base, df_test = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df[['Domain', 'ECDF_FCR_Category']]
    )
    
    # Display distribution of test set
    distribution = df_test.groupby(['Domain', 'ECDF_FCR_Category']).size().reset_index(name='Count')
    print("\n📊 Distribution of Papers in df_test by Domain and ECDF_FCR_Category:\n")
    print(distribution)
    
    return df_knowledge_base, df_test
