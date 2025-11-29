import os
import re
import pandas as pd
from collections import Counter
from scipy.stats import skew

def remove_descent_variants(word_list):
    pattern = re.compile(r'^[\[\(\{\"\']*descent[\]\)\}\"\'\.,!?;:]*$', re.IGNORECASE)
    cleaned_list = [word for word in word_list if not pattern.match(word)]
    return cleaned_list

language_resources = {
    "English": "High",
    "Arabic": "High",
    "Chinese": "High",
    "Bengali": "Medium",
    "French": "High",
    "Hausa": "Low",
    "Hindi": "High",
    "Japanese": "High",
    "Korean": "High",
    "Nepali": "Low",
    "Russian": "Medium",
    "Somali": "Low",
    "Spanish": "High",   
    "Swahili": "Low",    
    "Vietnamese": "Medium"
}

def process_csv_file(filename):

    df = pd.read_csv(filename)
    column_name = df.columns[0]
    data_list = df[column_name].astype(str).tolist() 

    cleaned_words = remove_descent_variants(data_list)

    freq_counter = Counter(cleaned_words)
    frequencies = list(freq_counter.values())

    if len(frequencies) < 3:

        return None
    return skew(frequencies)

def main():

    csv_dir = "./"  
    skew_per_language = {}
    
    for filename in os.listdir(csv_dir):
        if not filename.endswith(".csv"):
            continue
        
        lang = filename.split('_')[0].split('.')[0]
        if lang not in language_resources:
            print(f"[Warning] Language '{lang}' not in resource mapping. Skipping {filename}")
            continue
        
        filepath = os.path.join(csv_dir, filename)
        sk = process_csv_file(filepath)
        if sk is not None:
            skew_per_language[lang] = sk
        else:
            print(f"[Warning] Not enough data for skewness in {filename}")

    resource_groups = {"High": [], "Medium": [], "Low": []}
    for lang, sk_val in skew_per_language.items():
        resource_level = language_resources.get(lang)
        if resource_level:
            resource_groups[resource_level].append(sk_val)

    for level, values in resource_groups.items():
        if values:
            avg_skew = sum(values) / len(values)
            print(f"Average skewness for {level} resource languages: {avg_skew:.4f} (n={len(values)})")
        else:
            print(f"No data for {level} resource languages.")

if __name__ == "__main__":
    main()
