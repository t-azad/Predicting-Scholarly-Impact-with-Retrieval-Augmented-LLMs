from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, ndcg_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import re
import concurrent.futures
import threading

# ✅ Parameters
NUM_SAMPLES = 5  # Self-consistency: Number of times to call LLM per paper
NUM_WORKERS = 4  # Parallel processing: Number of LLM calls at once
processed_count = 0  # Thread-safe counter
counter_lock = threading.Lock()  # Prevents race conditions in multithreading


# ✅ Define Prompt Template using LangChain
prompt_template = PromptTemplate(
    input_variables=["query_paper"],
    template=(
        "Given a new research paper, predict its normalized FCR score between 0 and 1, "
        "where 0 means very low impact and 1 means very high impact.\n\n"
        "**🔹 New Research Paper:**\n"
        "{query_paper}\n\n"
        "**Return only a number. Do not add explanations or text.**"
    )
)

# ✅ Update Counter Function
def update_counter(total):
    global processed_count
    with counter_lock:
        processed_count += 1
        print(f"🔹 Processed {processed_count}/{total} papers", end="\r")

# ✅ Create Prompt
def create_prompt(query_paper, selected_features):
    query_content = "\n".join([f"{feature}: {query_paper[feature]}" for feature in selected_features])
    return prompt_template.format(query_paper=query_content)

# ✅ Generate Prediction with Dynamic Model Initialization
def zero_shot_fcr_prediction(query_paper, selected_features, model_name="llama3"):
    """Predicts the normalized FCR score using Zero-Shot Prompting with Self-Consistency."""
    prompt = create_prompt(query_paper, selected_features)
    scores = []

    # Initialize LLM dynamically based on the model name
    llm = Ollama(model=model_name)
    print("\n--------------------------\n")
    print(prompt)
    print("\n--------------------------\n")
    
    for _ in range(NUM_SAMPLES):
        response = llm.invoke(prompt)
        response_text = response.strip()
        match = re.search(r"\d+\.\d+", response_text)
        score = float(match.group(0)) if match else np.nan
        scores.append(score)

    return np.nanmedian(scores)

# ✅ Process Single Test Paper
def process_test_paper(row, selected_features, model_name="llama3", total_papers=1):
    predicted_fcr = zero_shot_fcr_prediction(row, selected_features, model_name)
    update_counter(total_papers)

    return {
        "Model": model_name,
        "Title": row["Title"],
        "Abstract": row["Abstract"],
        "Domain": row["Domain"],
        "Feature_Set": str(selected_features),
        "Actual_FCR": row["ECDF_FCR"],
        "Predicted_FCR": predicted_fcr,
        "Flesch Reading Ease": row["Flesch Reading Ease"],
        "Gunning Fog Index": row["Gunning Fog Index"],
        "Year": row["PubYear"]
    }

# ✅ Run Experiment with Multiple Models
def run_zero_shot_experiment(df_test, feature_sets, filename_prediction, filename_evaluation, model_name="llama3"):
    global processed_count
    processed_count = 0
    total_papers = len(df_test)

    all_predictions = []
    all_eval_results = []

    for feature_set in feature_sets:
        print(f"\n🚀 Running Zero-Shot Prompting with Features: {feature_set} using {model_name}")

        # ✅ Parallel Processing
        predictions_list = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {
                executor.submit(process_test_paper, row, feature_set, model_name, total_papers): row 
                for _, row in df_test.iterrows()
            }
            for future in concurrent.futures.as_completed(futures):
                predictions_list.append(future.result())

        df_predictions = pd.DataFrame(predictions_list)
        all_predictions.append(df_predictions)

        # ✅ Compute Evaluation Metrics
        mae = round(mean_absolute_error(df_predictions["Actual_FCR"], df_predictions["Predicted_FCR"]), 3)
        mse = round(mean_squared_error(df_predictions["Actual_FCR"], df_predictions["Predicted_FCR"]), 3)
        rmse = round(np.sqrt(mse), 3)
        r2 = round(r2_score(df_predictions["Actual_FCR"], df_predictions["Predicted_FCR"]), 3)
        
        true_relevance = df_predictions["Actual_FCR"].values.reshape(1, -1)
        predicted_relevance = df_predictions["Predicted_FCR"].values.reshape(1, -1)
        ndcg = round(ndcg_score(true_relevance, predicted_relevance), 3)

        eval_results = {
            "Model": model_name,
            "Feature_Set": str(feature_set),
            "MAE": mae,
            "NDCG": ndcg,
            "MSE": mse,
            "RMSE": rmse,
            "R²": r2,
        }
        all_eval_results.append(eval_results)

        print(f"\n📊 Evaluation Metrics for Feature Set: {feature_set}")
        print(f"   - MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, NDCG: {ndcg:.4f}")

    # ✅ Save Results
    pd.concat(all_predictions, ignore_index=True).to_csv(filename_prediction, index=False)
    pd.DataFrame(all_eval_results).to_csv(filename_evaluation, index=False)
    print(f"\n✅ All Experiments Completed! Results saved to {filename_prediction} and {filename_evaluation}")

# ✅ Create Dataset Function
def create_dataset(dataset, n_samples_per_category, test_size=0.1, random_state=42):
    df = pd.read_csv(dataset)
    df = df.groupby("ECDF_FCR_Category", group_keys=False).apply(
        lambda x: x.sample(n=n_samples_per_category, random_state=random_state)
    )
    df_knowledge_base, df_test = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df[['Domain', 'ECDF_FCR_Category']]
    )

    # ✅ Show Test Set Distribution
    distribution = df_test.groupby(['Domain', 'ECDF_FCR_Category']).size().reset_index(name='Count')
    print("\n📊 Distribution of Papers in df_test by Domain and ECDF_FCR_Category:\n")
    print(distribution)

    return df_knowledge_base, df_test
