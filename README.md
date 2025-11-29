# CLINIC : Evaluating Multilingual Trustworthiness in Language Models for Healthcare

![Model](https://img.shields.io/badge/Model-GPT-4o-mini-9AC75F)
![Model](https://img.shields.io/badge/Model-Gemini-1.5-Flash-9AC75F)
![Model](https://img.shields.io/badge/Model-Gemini-2.5-Pro-9AC75F)
![Model](https://img.shields.io/badge/Model-OpenBioLLM-8B-9AC75F)
![Model](https://img.shields.io/badge/Model-UltraMedical-9AC75F)
![Model](https://img.shields.io/badge/Model-MMedLLamal-9AC75F)
![Model](https://img.shields.io/badge/Model-LLaMA-3.2-3B-9AC75F)
![Model](https://img.shields.io/badge/Model-Qwen-2-1.5B-9AC75F)
![Model](https://img.shields.io/badge/Model-Phi-4mini-9AC75F)
![Model](https://img.shields.io/badge/Model-Qwen3-32B-9AC75F)
![Model](https://img.shields.io/badge/Model-DeepSeek-R1-9AC75F)
![Model](https://img.shields.io/badge/Model-DeepSeek-R1-LLaMA-9AC75F)
![Model](https://img.shields.io/badge/Model-QwQ-32B-9AC75F)

![Dataset](https://img.shields.io/badge/Dataset-CLINIC-4C84D3)

![Task](https://img.shields.io/badge/Task-Truthfulness-C55C53)
![Task](https://img.shields.io/badge/Task-Robustness-C55C53)
![Task](https://img.shields.io/badge/Task-Fairness-C55C53)
![Task](https://img.shields.io/badge/Task-Safety-C55C53)
![Task](https://img.shields.io/badge/Task-Privacy-C55C53)


First ever multilingual benchmark for trustworthiness in healthcare.

## Abstract

Integrating language models (LMs) in healthcare systems holds great promise for improving medical workflows and decision-making. However, a critical barrier to their real-world adoption is the lack of reliable evaluation of their trustworthiness, especially in multilingual healthcare settings. Existing LMs are predominantly trained in high-resource languages, making them ill-equipped to handle the complexity and diversity of healthcare queries in mid- and low-resource languages, posing significant challenges for deploying them in global healthcare contexts where linguistic diversity is key. In this work, we present CLINIC, a Comprehensive Multilingual Benchmark to evaluate the trustworthiness of language models in healthcare. CLINIC systematically benchmarks LMs across five key dimensions of trustworthiness: truthfulness, fairness, safety, robustness, and privacy, operationalized through 18 diverse tasks, spanning 15 languages (covering all the major continents), and encompassing a wide array of critical healthcare topics like disease conditions, preventive actions, diagnostic tests, treatments, surgeries, and medications. Our extensive evaluation reveals that LMs struggle with factual correctness,demonstrate bias across demographic and linguistic groups, and are susceptible to privacy breaches and adversarial attacks. By highlighting these shortcomings, CLINIC lays the foundation for enhancing the global reach and safety of LMs in healthcare across diverse languages.

![](image.png)


## 📌 Key contributions of our work

1. Comprehensive Multidimensional Evaluation: We establish a structured trustworthiness evaluation framework covering truthfulness, fairness, safety, privacy, and robustness through 18 sub-tasks– adversarial attacks, consistency verification, disparagement, exaggerated safety, stereotype and preference fairness, hallucination, honesty, jailbreak and OoD robustness, privacy leakage, toxicity and sycophancy.
2. Domain-Specific Healthcare Coverage: CLINIC offers 28,800 carefully curated samples from six key healthcare domains, including patient conditions, preventive healthcare, diagnostics and laboratory tests, pharmacology and medication, surgical and procedural treatment, and emergency medicine.
3. Global Linguistic Coverage: CLINIC supports 15 languages from diverse regions, including Asia, Africa, Europe, and the America, ensuring broad cultural and linguistic representation.
4. Extensive Model Benchmarking: We conduct a comprehensive evaluation of 13 language models, including small and large open-weight, medical, and reasoning models, providing a holistic analysis of language models across varied healthcare scenarios.
5. Expert Validation: All evaluation tasks and their respective criteria have been validated and refined in consultation with healthcare domain experts, ensuring clinical accuracy and real-world relevance.

<img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/f728b5af-b6db-4730-8d57-35147f6b8bcc" />


## CLINIC vs Other Benchmarks



## Dataset

### Data Collection

We used MedlinePlus (NLM, 2025) as our primary data source because it provides broad coverage of medical subdomains and high-quality English and professionally translated multilingual content. Unlike prior datasets (Wang et al., 2024; Qiu et al., 2024), it includes low-resource and geographically diverse languages with clinically vetted translations. To support out-of-distribution evaluation and ensure current medication information, we additionally incorporated FDA drug documents with available parallel multilingual versions.

### Dataset Dimensions

<!-- <img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/aba4c61b-a129-4c81-9900-5feb7c09b8ae" /> -->
<img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/657c206e-6ebf-47b6-a5e7-593dddee671d" />



### Construction of CLINIC

<img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/7e20351d-a9ab-44b9-8d5d-842f7276d3ad" />
Step 1 involves data collection and mapping English samples to their corresponding multilingual versions. Step 2 applied a two-step prompting strategy to generate
additional samples. Step 3 focused on sample validation to determine final inclusion in CLINIC.

### Data Statistics

1. Distribution of samples across different dimensions of CLINIC
<img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/8a501d88-4714-4a84-8f61-7793ddb69c4e" />

1. Distribution of samples across subdomains, where some samples fall under multiple categories.
<img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/d528b327-924f-4e3b-a456-b494aec5f389" />



## Performance

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for model inference)
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/AikyamLab/clinic
cd clinic
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note:** For CUDA support with PyTorch, you may need to install PyTorch separately based on your CUDA version. Visit [PyTorch's official website](https://pytorch.org/get-started/locally/) for installation instructions.

## Contents of this repo

This repository contains the model generation and response evaluation scripts used in the CLINIC benchmark. The repository is organized as follows:

- **`generation/`**: Contains 16 sub-folders, each with a Python script for generating model responses for different tasks
- **`evaluation/`**: Contains 16 sub-folders, each with a Python script for evaluating model responses

**Note:** The paper contains 18 tasks. During response generation and evaluation, we combined three tasks - False Confidence Test (FCT), False Question Test (FQT), and None of the Above Test (NOTA) - resulting in 16 scripts each for generation and evaluation.

## Usage

### Generation Scripts

Each generation script in the `generation/` folder follows a similar structure:

1. Configure the `model_path` variable with your model path
2. Set the `model_name` variable
3. Ensure the corresponding dataset CSV file is in the script's directory
4. Run the script:
```bash
cd generation/<task-name>/
python <script-name>.py
```

### Evaluation Scripts

Each evaluation script in the `evaluation/` folder:

1. Configure any required API keys (e.g. OpenAI API key, Perspective API key for toxicity evaluation)
2. Set the `MODEL_NAME` variable to match the model you're evaluating
3. Ensure response files are in the expected directory structure
4. Run the script:
```bash
cd evaluation/<task-name>/
python <task-name>_eval.py
```

### Available Tasks

The repository includes scripts for the following tasks:

- **Truthfulness**: Hallucinations, Honesty, Out-of-Domain (OOD)
- **Fairness**: Fairness-Preference, Fairness-Stereotype
- **Safety**: Toxicity, Disparagement, Exaggerated Safety
- **Robustness**: Adversarial Attacks, Consistency, Colloquial, Jailbreak-DAN, Jailbreak-PAIRS
- **Privacy**: Privacy
- **Sycophancy**: Sycophancy-Persona, Sycophancy-Preference

## Requirements

See `requirements.txt` for the complete list of dependencies. Key dependencies include:

- `transformers`: For loading and running language models
- `torch`: PyTorch for deep learning operations
- `pandas`: For data manipulation
- `openai`: For OpenAI API-based evaluations
- `requests`: For API calls (e.g., Perspective API)
- `scipy`: For statistical computations
- `FlagEmbedding`: For embedding-based evaluations

## Citation

If you use CLINIC benchmark in your research, please cite our repo:

```bibtex
@misc{githubrepo,
  author       = {Aikyam Lab},
  title        = {clinic},
  howpublished = {\url{https://github.com/AikyamLab/clinic}},
  year         = {2025},
  note         = {Version 1.0}
}

```

## License

[Add your license information here]

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
