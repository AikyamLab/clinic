import os
import pandas as pd
import json
import re
from FlagEmbedding import BGEM3FlagModel

def clean_response(response):
    if '<think>' in response and '</think>' in response:
        response = response.split('</think>')[-1]
    response = re.sub(r'^\s+|\s+$', '', response)
    return response

def load_emb_model():
    return BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

def get_embeddings(response, model):
    if isinstance(response, float):
        response = str(response)
    if not isinstance(response, str):
        raise ValueError(f"Invalid input to get_embeddings (not a string): {response}")
    response = [response.strip()]
    return model.encode(response, batch_size=12, max_length=8192)['dense_vecs']

def get_similarity_score(em1, em2):
    return em1 @ em2.T

def process_adv_files(model_name, task_name, ground_truth_path):
    base_path = os.getcwd()
    response_root = os.path.join(base_path, f"{model_name}/{task_name}_respon_{model_name}")
    ground_truth_df = pd.read_csv(ground_truth_path)

    adv_files = ["q3", "q4", "q5", "q12", "q13"]
    model = load_emb_model()

    avg_scores_by_q = {q: [] for q in adv_files}

    for language in os.listdir(response_root):
        lang_path = os.path.join(response_root, language)
        if not os.path.isdir(lang_path):
            continue

        print(f"[Info] Processing language: {language}")
        
        for q in adv_files:
            fname = f"{language}_adv_{q}_{model_name}.csv"
            adv_file_path = os.path.join(lang_path, fname)
            if not os.path.exists(adv_file_path):
                print(f"[Warning] {fname} not found. Skipping...")
                continue
            
            print(f"[Info] Evaluating file: {fname}")
            df = pd.read_csv(adv_file_path)
            df = df[:100]

            scores = []
            for idx, row in df.iterrows():
                original = ground_truth_df.at[idx, "original_question"]
                modified = row["Response"]

                try:
                    em1 = get_embeddings(original, model)
                    em2 = get_embeddings(modified, model)
                    score = get_similarity_score(em1, em2)
                    scores.append(float(score.squeeze()))
                except Exception as e:
                    print(f"[Warning] Error at idx {idx}: {e}")
                    scores.append(0.0)

            df['similarity_score'] = scores
            output_path = os.path.join(base_path, f"{model_name}_{language}_adv_{q}_results.csv")
            df.to_csv(output_path, index=False)
            print(f"[Info] Saved: {output_path}")

            avg_score = sum(scores) / len(scores) if scores else 0.0
            avg_scores_by_q[q].append({
                "language": language,
                "average_score": round(avg_score, 4),
                "model": model_name
            })

    for q in adv_files:
        avg_json_path = os.path.join(base_path, f"{model_name}_adv_{q}_averages.json")
        with open(avg_json_path, "w") as f:
            json.dump(avg_scores_by_q[q], f, indent=2)
        print(f"[Info] Saved average score JSON: {avg_json_path}")

if __name__ == "__main__":
    MODEL_NAME = ""
    TASK_NAME = "adversarial-attacks"
    GROUND_TRUTH_PATH = ""  # update with actual path
    process_adv_files(MODEL_NAME, TASK_NAME, GROUND_TRUTH_PATH)
