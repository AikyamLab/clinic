import os
import pandas as pd
import json
from collections import Counter
import re
from openai import OpenAI, RateLimitError
import time
from FlagEmbedding import BGEM3FlagModel

def clean_response(response):
    response = re.sub(r'[^\x00-\x7F]+', '', response)  
    if '<think>' in response and '</think>' in response:
        response = response.split('</think>')[-1]   
    response = re.sub(r'^\s+|\s+$', '', response)
    return response

def load_emb_model(path = 'BAAI/bge-m3'):
    model = BGEM3FlagModel('BAAI/bge-m3',  
                       use_fp16=True) 
    return model

def get_embeddings(response, model):
    
    embeddings_1 = model.encode(response, 
                            batch_size=12, 
                            max_length=8192, 
                            )['dense_vecs']
    return embeddings_1

def get_similarity_score(em1, em2):
    similarity = em1@ em2.T
    return similarity
            
def get_score(model,og, mod):
    
    org = get_embeddings(og, model)
    modi = get_embeddings(mod, model)

    score = get_similarity_score(org, modi)
    return score



def evaluate_model(model_name):
    """
    Args:
        model_folder_name: str, e.g., "phi4-mini-toxicity"
        model_name: str, like "phi4-mini"
        input_csv_name: str, name of CSV file containing language queries
    """
    
    current_dir = os.getcwd()

    responses_root = os.path.join(current_dir, f"{model_name}/consistency_respon_{model_name}") 
    

    if not os.path.exists(responses_root):
        raise FileNotFoundError(f"Responses folder {responses_root} not found. Please check!")
    results = []

    language_folders = [
        os.path.join(responses_root, lang_folder)
        for lang_folder in os.listdir(responses_root)
        if os.path.isdir(os.path.join(responses_root, lang_folder))
    ]
    # print(language_folders)

    model = load_emb_model()
    for lang_folder in language_folders:
        lang = os.path.basename(lang_folder)
        # print(lang)
        csv_filename_og = f"{lang}_original_{model_name}.csv"
        csv_filename_mod = f"{lang}_modified_{model_name}.csv"
        csv_path_og = os.path.join(lang_folder, csv_filename_og)
        csv_path_mod = os.path.join(lang_folder, csv_filename_mod)
        if not os.path.exists(csv_path_og):
            print(f"[Warning] {csv_filename_og} not found inside {lang_folder}. Skipping...")
            continue

        print(f"[Info] Evaluating {csv_filename_og}...")
        
        if not os.path.exists(csv_path_mod):
            print(f"[Warning] {csv_filename_mod} not found inside {lang_folder}. Skipping...")
            continue

        print(f"[Info] Evaluating {csv_filename_mod}...")
        model_df_og = pd.read_csv(csv_path_og)
        model_df_mod = pd.read_csv(csv_path_mod)
        model_df_mod = model_df_mod[:100]

        consistency_score = 0
        total_samples = 0

        scores = []

        for idx, row in model_df_mod.iterrows():
            if idx>=100: 
                continue
            original = model_df_og.at[idx, "Response"]  
            modified = row['Response']     
            
            score = get_score(model,original, modified)
            scores.append(score)
            total_samples+=1

        consistency_score = sum(scores)
        avg_consistency_score = consistency_score / total_samples if total_samples > 0 else 0

        model_df_mod['score'] = scores
        output_csv_path = os.path.join(current_dir, f"{model_name}_{lang}_consistency_results.csv")
        model_df_mod.to_csv(output_csv_path, index=False)

        results.append({
            "model": model_name,
            "task": "consistency",
            "language": lang,
            "consistency_score_sum": consistency_score,
            "total_samples": total_samples,
            "avg consistency score" : avg_consistency_score
        })

    output_json_path = os.path.join(current_dir, f"{model_name}_consistency_results.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\nEvaluation complete. Results saved at {output_json_path}")


if __name__ == "__main__":

    MODEL_NAME = ""                
    evaluate_model(MODEL_NAME)
