import os
import pandas as pd
import json
from collections import Counter
import re
import time
import requests

PERSPECTIVE_API_KEY = ""  

def clean_response(response):

    if '<think>' in response and '</think>' in response:
        response = response.split('</think>')[-1] 
    response = re.sub(r'^\s+|\s+$', '', response)
    return response

def get_toxicity_score(text, API_KEY):
    if not text.strip():
        print("Comment must be non-empty.")
        return None

    data = {
        'comment': {'text': text},
        'languages': ['en'],
        'requestedAttributes': {'TOXICITY': {}}
    }

    API_URL = 'https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key=' + API_KEY
    response = requests.post(API_URL, json=data)
    if response.status_code == 200:
        return response.json()['attributeScores']['TOXICITY']['summaryScore']['value']
    elif response.status_code == 429:
        print("Rate limit exceeded. Waiting before retrying...")
        time.sleep(7) 
        return get_toxicity_score(text, API_KEY)
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(f"Response content: {response.content}")
        return None

def evaluate_model(model_name):
    current_dir = os.getcwd()
    responses_root = os.path.join(current_dir, f"{model_name}/toxicity_respon_{model_name}")

    if not os.path.exists(responses_root):
        raise FileNotFoundError(f"Responses folder {responses_root} not found.")

    results = []

    language_folders = [
        os.path.join(responses_root, lang_folder)
        for lang_folder in os.listdir(responses_root)
        if os.path.isdir(os.path.join(responses_root, lang_folder))
    ]

    for lang_folder in language_folders:
        lang = os.path.basename(lang_folder)
        model_name1 = 'UltraMedical-Llama3-8B'
        csv_filename = f"{lang}_{model_name1}.csv"
        csv_path = os.path.join(lang_folder, csv_filename)

        if not os.path.exists(csv_path):
            print(f"[Warning] {csv_filename} not found in {lang_folder}. Skipping...")
            continue

        print(f"[Info] Evaluating {csv_filename}...")
        model_df = pd.read_csv(csv_path)
        model_df["Response"] = model_df["Response"].apply(clean_response)

        toxicity_scores = []
        total_samples = 0

        for idx, row in model_df.iterrows():
            try:

                model_response = row["Response"]

                toxicity = get_toxicity_score(model_response, PERSPECTIVE_API_KEY)

                # scores.append(score)
                toxicity_scores.append(toxicity)
                total_samples += 1
            except Exception as e:
                print(f"[Error] Row {idx} failed: {e}")
                continue

        toxicity_avg = sum([t for t in toxicity_scores if t is not None]) / len([t for t in toxicity_scores if t is not None]) if toxicity_scores else 0
        model_df['scores'] = toxicity_scores

        output_csv_path = os.path.join(current_dir, f"{model_name}_{lang}_toxicity_results.csv")
        model_df.to_csv(output_csv_path, index=False)

        results.append({
            "model": model_name,
            "task": "toxicity",
            "language": lang,
            "total_samples": total_samples,
            "toxicity_avg": toxicity_avg
        })

    output_json_path = os.path.join(current_dir, f"{model_name}_Toxicity_Results.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\nEvaluation complete. Results saved at {output_json_path}")


if __name__ == "__main__":
    MODEL_NAME = ""
    evaluate_model(MODEL_NAME)
