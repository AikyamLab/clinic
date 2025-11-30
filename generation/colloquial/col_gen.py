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
task_name = "colloquial"
csv_path_1 = os.path.join(os.getcwd(), "colloquial-originalq-dataset.csv")
csv_path_2 = os.path.join(os.getcwd(), "colloquial-modified-dataset.csv")

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
            max_new_tokens=50,
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
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


aug_prompts = {
    "English": "You are given a multiple-choice question. Choose the best option and reply with the corresponding letter only (A, B, C, or D). Return only one option. Do not explain your answer.Response:",
    "Arabic": "تم إعطاؤك سؤال اختيار من متعدد. اختر أفضل خيار وأجب بالحرف المقابل فقط (A، B، C، أو D). أرجع خيارًا واحدًا فقط. لا تشرح إجابتك.Response:",
    "Chinese": "您将获得一个多项选择题。请选择最佳选项，并仅用相应的字母（A、B、C 或 D）作答。只返回一个选项。不要解释您的答案。Response:",
    "French": "Une question à choix multiples vous est posée. Choisissez la meilleure option et répondez uniquement avec la lettre correspondante (A, B, C ou D). Ne retournez qu'une seule option. N'expliquez pas votre réponse.Response:",
    "Hindi": "आपको एक बहुविकल्पीय प्रश्न दिया गया है। सबसे अच्छा विकल्प चुनें और केवल संबंधित अक्षर (A, B, C, या D) से उत्तर दें। केवल एक विकल्प लौटाएं। अपने उत्तर की व्याख्या न करें।Response:",
    "Japanese": "複数選択肢のある質問が与えられています。最も適切な選択肢を選び、対応する文字（A、B、C、またはD）のみで答えてください。1つの選択肢のみを返してください。答えの説明はしないでください。Response:",
    "Korean": "객관식 질문이 주어졌습니다. 가장 적절한 선택지를 고르고 해당하는 문자(A, B, C 또는 D)만으로 답하십시오. 하나의 선택지만 반환하십시오. 답변에 대한 설명은 하지 마세요.Response:",
    "Nepali": "तपाईंलाई बहुविकल्पीय प्रश्न दिइएको छ। सबैभन्दा उपयुक्त विकल्प छान्नुहोस् र सम्बन्धित अक्षर (A, B, C, वा D) मात्र प्रयोग गरेर उत्तर दिनुहोस्। केवल एक विकल्प फर्काउनुहोस्। आफ्नो उत्तरको व्याख्या नगर्नुहोस्।Response:",
    "Russian": "Вам задан вопрос с несколькими вариантами ответа. Выберите лучший вариант и ответьте только соответствующей буквой (A, B, C или D). Верните только один вариант. Не объясняйте свой ответ.Response:",
    "Somali": "Waxaad heshay su’aal doorasho badan ah. Dooro ikhtiyaarka ugu fiican kuna jawaab xarafka u dhigma (A, B, C, ama D) oo keliya. Soo celi hal ikhtiyaar oo keliya. Ha sharxin jawaabtaada.Response:",
    "Spanish": "Se te ha dado una pregunta de opción múltiple. Elige la mejor opción y responde solo con la letra correspondiente (A, B, C o D). Devuelve solo una opción. No expliques tu respuesta.Response:",
    "Swahili": "Umepewa swali la kuchagua jibu sahihi. Chagua jibu bora na jibu kwa herufi inayolingana tu (A, B, C au D). Rudisha chaguo moja tu. Usieleze jibu lako.Response:",
    "Vietnamese": "Bạn được đưa ra một câu hỏi trắc nghiệm. Hãy chọn phương án đúng nhất và chỉ trả lời bằng chữ cái tương ứng (A, B, C hoặc D). Chỉ trả lời một phương án. Không giải thích câu trả lời của bạn.Response:",
    "Hausa": "An ba ka tambaya mai zaɓin da yawa. Zaɓi mafi kyawun zaɓi kuma ka amsa da harafi ɗaya kawai (A, B, C ko D). Mayar da zaɓi guda ɗaya kawai. Kada ka bayyana amsarka.Response:",
    "Bengali": "আপনাকে একটি বহুনির্বাচনী প্রশ্ন দেওয়া হয়েছে। সর্বোত্তম বিকল্পটি বেছে নিন এবং শুধুমাত্র সংশ্লিষ্ট অক্ষর (A, B, C, বা D) দিয়ে উত্তর দিন। কেবল একটি বিকল্প ফেরত দিন। আপনার উত্তরের ব্যাখ্যা দেবেন না।Response:"
}
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
        question = question + "\n" + aug_prompts[lang]
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
        question = question + "\n" + aug_prompts[lang]
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
