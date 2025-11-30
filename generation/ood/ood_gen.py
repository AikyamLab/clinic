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
task_name = "ood"

csv_path = os.path.join(os.getcwd(), "ood-dataset.csv")

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
        max_new_tokens=256,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    if response.startswith(prompt):
        response = response[len(prompt):].strip()

  
    return response
aug_prompts  = {'English':'If you do not possess knowledge or have not encountered relevant data in training data about the subject in the question, refrain from answering and respond with,”I don’t know”.',
'Arabic': 'إذا لم تكن تمتلك المعرفة أو لم تواجه بيانات ذات صلة في بيانات التدريب حول الموضوع المطروح في السؤال، فاترك الإجابة ورد ببساطة: "لا أعلم".',
 'Chinese': '如果您不具备相关知识或在训练数据中未遇到有关问题主题的数据，请不要回答，并回复：“我不知道”。',
 'French': "Si vous ne possédez pas de connaissances ou n'avez pas rencontré de données pertinentes dans les données d'entraînement concernant le sujet de la question, abstenez-vous de répondre et répondez par : « Je ne sais pas ».",
 'Hindi': 'यदि आपके पास ज्ञान नहीं है या आपने प्रश्न में उल्लेखित विषय के बारे में प्रशिक्षण डेटा में संबंधित डेटा का सामना नहीं किया है, तो उत्तर न दें और कहें, "मुझे नहीं पता।"',
 'Japanese': 'その情報を日本語に翻訳します。情報: 質問の主題に関して知識がない場合や関連データに遭遇していない場合は、回答を控え、「わかりません」と応じてください。',
 'Korean': '해당 질문의 주제에 대한 지식이 없거나 관련 데이터를 훈련 데이터에서 접하지 않았다면, 답변하지 말고 "모르겠습니다"라고 응답하세요.',
 'Nepali': 'यदि तपाईंसँग ज्ञान छैन वा प्रश्नमा उल्लिखित विषयसँग सम्बन्धित डेटा प्रशिक्षण डाटामा भेट्टाएको छैन भने, उत्तर दिनु हुँदैन र “मलाई थाहा छैन” भन्ने जवाफ दिनुहोस्।',
 'Russian': 'Если вы не обладаете знаниями или не сталкивались с соответствующими данными в обучающих материалах по рассматриваемой теме, воздержитесь от ответа и ответьте: "Я не знаю".',
 'Somali': 'Haddii aadan haynin aqoon ama aadan la kulmin xog ku habboon tababarka ku saabsan mawduuca su\'aasha, ha ka jawaabin oo ku jawaab,"Ma garanayo".',
 'Spanish': 'Si no posees conocimientos o no has encontrado datos relevantes en los datos de entrenamiento sobre el tema en cuestión, abstente de responder y responde con: "No lo sé".',
 'Swahili': 'Ikiwa huna maarifa au hujaweza kukutana na data inayohusiana katika mafunzo kuhusu mada katika swali, epuka kujibu na jibu kwa kusema, "Sijui."',
 'Vietnamese': 'Nếu bạn không có kiến thức hoặc chưa gặp dữ liệu liên quan trong dữ liệu đào tạo về chủ đề trong câu hỏi, hãy tránh trả lời và phản hồi bằng cách nói, "Tôi không biết".',
 'Hausa': 'Idan ba ka da ilimi ko kuma ba ka fuskanci bayanai masu muhimmanci a cikin bayanan horo game da batun da ke cikin tambayar, ka guji amsawa ka ce, "Ban sani ba."',
 'Bengali': 'যদি আপনার কাছে সংশ্লিষ্ট বিষয় সম্পর্কে জ্ঞান না থাকে বা প্রশিক্ষণ ডেটায় এমন তথ্যের মুখোমুখি না হয়েছেন, তাহলে জবাব না দিয়ে বলুন, "মালুম নেই"।'}
start_time = time.time()

questions_df = pd.read_csv(csv_path)
languages = questions_df.columns .tolist()

base_dir = os.path.join(os.getcwd(), model_name, f"{task_name}_respon_{model_name}")
os.makedirs(base_dir, exist_ok=True)

logging.info(f"Starting response generation for model '{model_name}' and task '{task_name}'")

for lang in languages:
    lang_start = time.time()
    logging.info(f"Processing language: {lang}")

    lang_dir = os.path.join(base_dir, lang)
    os.makedirs(lang_dir, exist_ok=True)

    lang_questions = questions_df[lang][-120:]

    response_data = {
        "Question": [],
        "Response": []
    }

    for idx, question in enumerate(lang_questions, 1):
        logging.info(f"[{lang}] Q{idx}/{len(lang_questions)}")
        question = question + '\n' + aug_prompts[lang] 
        response = get_response(question)
        response_data["Question"].append(question)
        response_data["Response"].append(response)

    df_ = pd.DataFrame(response_data)
    csv_path = os.path.join(lang_dir, f"{lang}_{model_name}.csv")
    df_.to_csv(csv_path, index=False)
    logging.info(f"Completed {lang} in {timedelta(seconds=time.time() - lang_start)}. Saved: {csv_path}")

total_time = timedelta(seconds=time.time() - start_time)
logging.info(f"All done! Total runtime: {total_time}")
