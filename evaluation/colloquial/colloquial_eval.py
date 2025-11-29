import os
import pandas as pd
import json
from collections import Counter
import re
import string
import time

def clean_and_extract_choice(response):
    if not isinstance(response, str):
        return ""

    if '<think>' in response and '</think>' in response:
        response = response.split('</think>')[-1]
    response = response.translate(str.maketrans('', '', string.punctuation))
    response = re.sub(r'\s+', ' ', response).strip()
    match = re.search(r'\b([ABCD])\b', response)
    return match.group(1) if match else None

def get_score(resp, gt):
    return resp.lower() == gt.lower()

def evaluate_model(model_name, ground_truth_file):
    """
    Args:
        model_folder_name: str, e.g., "phi4-mini-toxicity"
        model_name: str, like "phi4-mini"
        ground_truth_file: str, name of CSV file containing language queries
    """

    current_dir = os.getcwd()
    responses_root = os.path.join(current_dir, f"{model_name}/colloquial_respon_{model_name}") 
    
    if not os.path.exists(responses_root):
        raise FileNotFoundError(f"Responses folder {responses_root} not found. Please check!")

    results = []

    language_folders = [
        os.path.join(responses_root, lang_folder)
        for lang_folder in os.listdir(responses_root)
        if os.path.isdir(os.path.join(responses_root, lang_folder))
    ]
    # print(language_folders)
    ground_truth_path = os.path.join(current_dir, ground_truth_file)
    if not os.path.exists(ground_truth_path):
        raise FileNotFoundError(f"Input CSV file {ground_truth_file} not found. Please check!")

    gt_df = pd.read_csv(ground_truth_path)

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
        model_df_og["Response"] = model_df_og["Response"].apply(clean_and_extract_choice)
        model_df_mod["Response"] = model_df_mod["Response"].apply(clean_and_extract_choice)

        acc_before = 0
        acc_after = 0
        total_samples = 0

        scores_before = []
        scores_after  = []

        for idx, row in model_df_og.iterrows():
            original = model_df_og.at[idx, "Response"]  
            modified = row['Response']    
            gt = gt_df.at[idx, "answer"]
            
            if original is None or original == "":
                continue
            score_bf = int(get_score(original, gt))
            scores_before.append(score_bf)
            score_af = int(get_score(modified, gt))
            scores_after.append(score_af)
            total_samples+=1

        acc_before = sum(scores_before)
        acc_after = sum(scores_after)
        avg_acc_after = acc_after / total_samples if total_samples > 0 else 0
        avg_acc_before = acc_before / total_samples if total_samples>0 else 0

        results.append({
            "model": model_name,
            "task": "colloquial",
            "language": lang,
            "acc_before":acc_before,
            "acc_after": acc_after,
            "avg_acc_after": avg_acc_after,
            "avg_acc_before" : avg_acc_before
        })

    output_json_path = os.path.join(current_dir, f"{model_name}_colloquial_results.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\nEvaluation complete. Results saved at {output_json_path}")


if __name__ == "__main__":

    ground_truth_file = ""  
    MODEL_NAME = ""                  

    evaluate_model(MODEL_NAME, ground_truth_file)
    
    
    
