import os
import pandas as pd
import logging
import time
from datetime import timedelta
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Environment setup
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Load model
model_path = ""
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("Model loaded successfully in 4-bit mode!\n")

# Configuration
model_name = ""
task_name = "consistency"
csv_path_1 = os.path.join(os.getcwd(), "consistency-original-dataset.csv")
csv_path_2 = os.path.join(os.getcwd(), "consistency-modified-dataset.csv")

# Logging setup
log_file = "response_generation.log"
log_path = os.path.join(os.getcwd(), log_file)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode='w'),
        logging.StreamHandler()
    ]
)

# Function to get model response
def get_response(question):
    prompt = question.strip()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
        **inputs,
        max_new_tokens=256,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    return response

# Load questions
questions_df = pd.read_csv(csv_path_1)
questions_moddf = pd.read_csv(csv_path_2)
languages = questions_df.columns.tolist()

# Create folder structure
base_dir = os.path.join(os.getcwd(), model_name, f"{task_name}_respon_{model_name}")
os.makedirs(base_dir, exist_ok=True)
logging.info(f"Starting response generation for model '{model_name}' and task '{task_name}'")

start_time = time.time()

# Process each language
for lang in languages:
    lang_start = time.time()
    logging.info(f"Processing language: {lang}")
    lang_dir = os.path.join(base_dir, lang)
    os.makedirs(lang_dir, exist_ok=True)

    # === Original Questions ===
    response_data_orig = {
        "Question": [],
        "Response": []
    }

    orig_questions = questions_df[lang][:100]
    for idx, question in enumerate(orig_questions, 1):
        logging.info(f"[{lang} | Original] Q{idx}/{len(orig_questions)}")
        response = get_response(question)
        response_data_orig["Question"].append(question)
        response_data_orig["Response"].append(response)

    df_orig = pd.DataFrame(response_data_orig)
    orig_csv_path = os.path.join(lang_dir, f"{lang}_original_{model_name}.csv")
    df_orig.to_csv(orig_csv_path, index=False)
    logging.info(f"Saved original responses: {orig_csv_path}")

    # === Modified Questions ===
    response_data_mod = {
        "Question": [],
        "Response": []
    }

    mod_questions = questions_moddf[lang][:100]
    for idx, question in enumerate(mod_questions, 1):
        logging.info(f"[{lang} | Modified] Q{idx}/{len(mod_questions)}")
        response = get_response(question)
        response_data_mod["Question"].append(question)
        response_data_mod["Response"].append(response)

    df_mod = pd.DataFrame(response_data_mod)
    mod_csv_path = os.path.join(lang_dir, f"{lang}_modified_{model_name}.csv")
    df_mod.to_csv(mod_csv_path, index=False)
    logging.info(f"Saved modified responses: {mod_csv_path}")

    logging.info(f"Completed {lang} in {timedelta(seconds=time.time() - lang_start)}")

# Summary
total_time = timedelta(seconds=time.time() - start_time)
logging.info(f"All done! Total runtime: {total_time}")
