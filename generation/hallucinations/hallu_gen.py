
import os
import pandas as pd
import logging
import time
from datetime import timedelta
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModel
import torch

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} \n")

model_path = ""

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto")

print("Model loaded successfully in 4-bit mode! \n")

model_name = ""
task_name = "hallucination"

csv_path = os.path.join(os.getcwd(), "hallucination-dataset.csv")

df = pd.read_csv(csv_path)

log_file = "response_generation.log"

log_path = os.path.join(os.getcwd(), log_file)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode='w'),  # 'w' to overwrite each time
        logging.StreamHandler()
    ]
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

def get_response(question):
    # You can optionally prepend system instructions or role-playing if needed
    prompt = question.strip()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
        **inputs,
        max_new_tokens=150,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
  
    return response

aug_prompts  = {'English': 'Choose one option from the options given to you, and return the chosen option.\n',
 'Arabic': 'اختر خيارًا واحدًا من الخيارات المقدمة لك، ثم أعد الخيار المختار.\n',
 'Chinese': '从给你的选项中选择一个选项，并返回所选的选项。\n',
 'French': "Choisissez une option parmi celles qui vous sont proposées et renvoyez l'option choisie.",
 'Hindi': 'केवल दिए गए विकल्पों में से एक विकल्प चुनें, और चुने गए विकल्प को लौटाएं। किसी और चीज़ के साथ प्रतिक्रिया न करें।',
 'Japanese': '提供されたオプションから 1 つを選択し、選択されたオプションを返します。\n',
 'Korean': '주어진 옵션 중 하나를 선택하고, 선택된 옵션을 반환하세요.',
 'Nepali': 'सिर्जना गरिएको विकल्पहरूमा केवल एक विकल्प चयन गर्नुहोस्, र चयन गरिएको विकल्प फर्काउनुहोस्। अन्य कुनै कुरासँग प्रतिक्रिया नदिनुहोस्।',
 'Russian': 'Выберите один вариант из предложенных вам вариантов и верните выбранный вариант.\n',
 'Somali': 'Ka dooro hal dookh xulashooyinka lagu siiyay, oo soo celi ikhtiyaarka la doortay.\n',
 'Spanish': 'Elige una opción de las opciones que se te ofrecen y devuelve la opción elegida.\n',
 'Swahili': 'Chagua chaguo moja kutoka kwa chaguo ulizopewa, na urudishe chaguo ulichochagua.\n',
 'Vietnamese': 'Chọn một tùy chọn từ các tùy chọn được cung cấp cho bạn và trả về tùy chọn đã chọn.\n',
 'Hausa': 'Zaɓi zaɓi ɗaya daga cikin zaɓuɓɓukan da aka ba ku, sannan ku mayar da zaɓin da aka zaɓa.\n',
 'Bengali': 'তোমাকে দেওয়া বিকল্পগুলি থেকে একটি বিকল্প বেছে নাও, এবং নির্বাচিত বিকল্পটি ফেরত দাও।\n'}

start_time = time.time()

questions_df = pd.read_csv(csv_path)
languages = questions_df.columns.tolist()

base_dir = os.path.join(os.getcwd(), model_name, f"{task_name}_respon_{model_name}")
os.makedirs(base_dir, exist_ok=True)

logging.info(f"Starting response generation for model '{model_name}' and task '{task_name}'")

for lang in languages:
    lang_start = time.time()
    logging.info(f"Processing language: {lang}")

    lang_dir = os.path.join(base_dir, lang)
    os.makedirs(lang_dir, exist_ok=True)

    lang_questions = questions_df[lang][:150]

    response_data = {
        "Question": [],
        "Response": []
    }

    for idx, question in enumerate(lang_questions, 1):
        logging.info(f"[{lang}] Q{idx}/{len(lang_questions)}")
        question = "Question: "+ "\n" + question + '\n' + aug_prompts[lang] +"\n" +"Answer: "
        response = get_response(question)
        response_data["Question"].append(question)
        response_data["Response"].append(response)

    df_ = pd.DataFrame(response_data)
    csv_path = os.path.join(lang_dir, f"{lang}_{model_name}.csv")
    df_.to_csv(csv_path, index=False)
    logging.info(f"Completed {lang} in {timedelta(seconds=time.time() - lang_start)}. Saved: {csv_path}")

total_time = timedelta(seconds=time.time() - start_time)
logging.info(f"All done! Total runtime: {total_time}")
