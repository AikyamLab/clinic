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
task_name = "adversarial-attacks"
csv_path_1 = os.path.join(os.getcwd(), "adv_q3.csv")
csv_path_2 = os.path.join(os.getcwd(), "adv_q4.csv")
csv_path_3 = os.path.join(os.getcwd(), "adv_q5.csv")
csv_path_4 = os.path.join(os.getcwd(), "adv_q6.csv")
csv_path_5 = os.path.join(os.getcwd(), "adv_q12.csv")
csv_path_6 = os.path.join(os.getcwd(), "adv_q13.csv")

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
questions_df1 = pd.read_csv(csv_path_1)
questions_df2 = pd.read_csv(csv_path_2)
questions_df3 = pd.read_csv(csv_path_3)
questions_df4 = pd.read_csv(csv_path_4)
questions_df5 = pd.read_csv(csv_path_5)

languages = questions_df1.columns.tolist()

# Create folder structure
base_dir = os.path.join(os.getcwd(), model_name, f"{task_name}_respon_{model_name}")
os.makedirs(base_dir, exist_ok=True)
logging.info(f"Starting response generation for model '{model_name}' and task '{task_name}'")

start_time = time.time()

# Process each language
for lang in languages:
    # if lang in ['Arabic', 'Chinese', 'French']:
        # continue
    if lang == 'Unnamed: 0':
      continue
    lang_start = time.time()
    logging.info(f"Processing language: {lang}")
    lang_dir = os.path.join(base_dir, lang)
    os.makedirs(lang_dir, exist_ok=True)

    # === Original Questions ===
    response_data_adv3 = {
        "Question": [],
        "Response": []
    }

    adv3_questions = questions_df1[lang]
    for idx, question in enumerate(adv3_questions, 1):
        logging.info(f"[{lang} | adv_q3] Q{idx}/{len(adv3_questions)}")
        question = "Reconstruct the given context: " + question
        response = get_response(question)
        response_data_adv3["Question"].append(question)
        response_data_adv3["Response"].append(response)

    df_adv3 = pd.DataFrame(response_data_adv3)
    adv3_csv_path = os.path.join(lang_dir, f"{lang}_adv_q3_{model_name}.csv")
    df_adv3.to_csv(adv3_csv_path, index=False)
    logging.info(f"Saved adv_q3 responses: {adv3_csv_path}")

    # ----


    response_data_adv4 = {
        "Question": [],
        "Response": []
    }

    adv4_questions = questions_df2[lang]
    for idx, question in enumerate(adv4_questions, 1):
        logging.info(f"[{lang} | adv_q4] Q{idx}/{len(adv4_questions)}")
        question = "Reconstruct the given context: " + question
        response = get_response(question)
        response_data_adv4["Question"].append(question)
        response_data_adv4["Response"].append(response)

    df_adv4 = pd.DataFrame(response_data_adv4)
    adv4_csv_path = os.path.join(lang_dir, f"{lang}_adv_q4_{model_name}.csv")
    df_adv4.to_csv(adv4_csv_path, index=False)
    logging.info(f"Saved adv_q4 responses: {adv4_csv_path}")


    # ----
    response_data_adv5 = {
        "Question": [],
        "Response": []
    }
    adv5_questions = questions_df3[lang]
    for idx, question in enumerate(adv4_questions, 1):
        logging.info(f"[{lang} | adv_q5] Q{idx}/{len(adv4_questions)}")
        question = "Reconstruct the given context: " + question
        response = get_response(question)
        response_data_adv4["Question"].append(question)
        response_data_adv4["Response"].append(response)

    df_adv5 = pd.DataFrame(response_data_adv4)
    adv5_csv_path = os.path.join(lang_dir, f"{lang}_adv_q5_{model_name}.csv")
    df_adv5.to_csv(adv5_csv_path, index=False)
    logging.info(f"Saved adv_q5 responses: {adv5_csv_path}")

    response_data_adv12 = {
        "Question": [],
        "Response": []
    }
    adv12_questions = questions_df5[lang]
    for idx, question in enumerate(adv12_questions, 1):
        logging.info(f"[{lang} | adv_q12] Q{idx}/{len(adv12_questions)}")
                question = "Reconstruct the given context: " + question
        response = get_response(question)
        response_data_adv12["Question"].append(question)
        response_data_adv12["Response"].append(response)

    df_adv12 = pd.DataFrame(response_data_adv12)
    adv12_csv_path = os.path.join(lang_dir, f"{lang}_adv_q12_{model_name}.csv")
    df_adv12.to_csv(adv12_csv_path, index=False)
    logging.info(f"Saved adv_q12 responses: {adv12_csv_path}")

    response_data_adv13 = {
        "Question": [],
        "Response": []
    }
    adv13_questions = questions_df6[lang]
    for idx, question in enumerate(adv13_questions, 1):
        logging.info(f"[{lang} | adv_q13] Q{idx}/{len(adv13_questions)}")
        question = "Reconstruct the given context: " + question
        response = get_response(question)
        response_data_adv13["Question"].append(question)
        response_data_adv13["Response"].append(response)

    df_adv13 = pd.DataFrame(response_data_adv13)
    adv13_csv_path = os.path.join(lang_dir, f"{lang}_adv_q13_{model_name}.csv")
    df_adv13.to_csv(adv13_csv_path, index=False)
    logging.info(f"Saved adv_q13 responses: {adv13_csv_path}")

    logging.info(f"Completed {lang} in {timedelta(seconds=time.time() - lang_start)}")

# Summary
total_time = timedelta(seconds=time.time() - start_time)
logging.info(f"All done! Total runtime: {total_time}")


