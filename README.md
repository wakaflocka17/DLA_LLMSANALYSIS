# 🧠 SENTIMENT ANALYSIS 2024/25 - UNICA

<p align="center">
  <a href="https://www.apache.org/licenses/LICENSE-2.0" target="_blank">
    <img src="https://img.shields.io/badge/License-Apache_2.0-4285F4?style=for-the-badge&logo=none&logoColor=white" alt="Apache License 2.0"/>
  </a>
  <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank">
    <img src="https://img.shields.io/badge/Docs-CC_BY_4.0-FBBC05?style=for-the-badge&logo=none&logoColor=white" alt="CC BY 4.0 License"/>
  </a>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/c437885d-d304-4714-b4c2-0f47409ed226" alt="sentiment" width="600">
</p>

<p align="center">
  Project on <b>Binary Sentiment Analysis</b> using <b>Pretrained</b>, <b>Fine-tuned</b> and <b>Ensemble</b> Transformer Models.
</p>

---

> ## 📑 Summary
> 01. [🧑🏻‍🎓 Student](#student)  
> 02. [📌 Description](#description)  
> 03. [📄 Notebooks Overview](#notebooks-overview)  
> 04. [📁 Project Structure](#project-structure)  
> 05. [🔐 Access to Hugging Face Models](#access-to-hugging-face-models)  
> 06. [🚀 Installation](#installation)  
> 07. [🧪 Run: Model Training & Evaluation](#run-model-training--evaluation)  
> 08. [📊 Metrics and Outputs](#metrics-and-outputs)  
> 09. [🖥️ Hardware and Limitations](#hardware-and-limitations)  
> 10. [🤝 Contributions](#contributions)  
> 11. [📝 Licenses](#licenses)  
> 12. [❓ How to Cite](#how-to-cite)

---

## 1. 🧑🏻‍🎓 Student <a name="student"></a>

#### Francesco Congiu  
> Student ID: 60/73/65300  
>  
>> E-Mail: f.congiu38@studenti.unica.it  

---

## 2. 📌 Description <a name="description"></a>
This project investigates the impact of fine-tuning transformer-based models on the **Sentiment Analysis** task using the **IMDb dataset**.  
Three architectures are explored:

1. **Decoder-Only**: GPT-Neo  
2. **Encoder-Only**: BERT  
3. **Encoder-Decoder**: BART  

Additionally, we evaluate the performance of an **ensemble strategy** via **majority voting**.  
Both pretrained and fine-tuned versions are evaluated to compare generalization capabilities.

---

## 3. 📄 Notebooks Overview  <a name="notebooks-overview"></a>
> [!NOTE]
> Each notebook is self-contained and was provided for reproducibility.
> 
> Below a quick overview of each file:

| Notebook | Purpose |
|----------|---------|
| `train_models_from_scratch.ipynb` | Fine-tune each model and evaluate them individually |
| `ensemble_model_evaluation.ipynb` | Run ensemble predictions with majority voting |
| `models_plots_and_results.ipynb` | *(Coming soon)* Visual analysis, calibration and fairness plots |

---

## 4. 📁 Project Structure <a name="project-structure"></a>

```plaintext
📦 sentiment-analysis-transformers/
├── 📁 data/                          # (optional: IMDb dataset if local)
├── 📁 experiments/
│   ├── 📁 plots/                     # Graphs and result plots
│   └── 📁 results/
│       ├── 📁 evaluation/
│       │   ├── 📁 finetuned/
│       │   │   ├── bart-base-imdb.json
│       │   │   ├── bert-base-uncased-imdb.json
│       │   │   └── gpt-neo-2.7b-imdb.json
│       │   └── 📁 pretrained/
│       │       ├── bart-base-imdb.json
│       │       ├── bert-base-uncased-imdb.json
│       │       └── gpt-neo-2.7b-imdb.json
│       └── 📁 validation/
│           └── 📁 finetuned/
│               ├── bart-base-imdb_metrics.json
│               ├── bert-base-uncased-imdb_metrics.json
│               └── gpt-neo-2.7b-imdb_metrics.json
│
├── 📁 models/                        # Folder for storing our models
├── 📁 notebooks/
│   ├── train_models_from_scratch.ipynb
│   ├── ensemble_model_evaluation.ipynb
│   └── plot_results_and_test_models.ipynb
│
├── 📁 src/
│   ├── 📁 architectures/
│   │   ├── model_bart_base_imdb.py
│   │   ├── model_bert_base_uncased_imdb.py
│   │   ├── model_gpt_neo_2_7b_imdb.py
│   │   └── model_ensemble_majority_voting.py
│   ├── aggregate_json.py
│   ├── data_preprocessing.py
│   ├── download_models.py
│   ├── ensemble_analysis.py
│   ├── evaluate.py
│   ├── evaluate_ensemble.py
│   ├── model_configs.py
│   ├── model_configs_eval.py
│   ├── model_factory.py
│   ├── plot_results.py
│   ├── train.py
│   ├── upload_models.py
│   └── utils.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 5. 🔐 Access to Hugging Face Models <a name="access-to-hugging-face-models"></a>

In order to download and use pretrained models from the 🤗 Hugging Face Hub (like `bert-base-uncased`, `gpt-neo-2.7B`, or `bart-base`), you’ll need to authenticate.

### 5.1 🪪 How to get your Hugging Face Token

1. Visit [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New Token**, choose role `Read` and generate it
3. Copy the token to your clipboard

When running the notebook, you’ll be prompted to enter your token via:
```python
from huggingface_hub import notebook_login
notebook_login()
```
> [!NOTE]
> Run this manually in the first cell of the notebook if not already included. You only need to do this once per environment or session.

---

## 6. 🚀 Installation <a name="installation"></a>
Install requirements for any notebook as needed. For local runs, Python ≥ 3.8 is required.
> [!NOTE]
> For each notebook, you can use a dedicated environment to keep dependencies isolated.

---

## 7. 🧪 Run: Model Training & Evaluation <a name="run-model-training--evaluation"></a>

### 7.1 📘 `train_models_from_scratch.ipynb`
The notebook performs the entire process of analyzing the performance of pre-trained and fine-tuned models on the **Sentiment Analysis** task with the IMDb dataset. The following are the main steps write in the first notebook:

#### 7.1.1 ⚙️ Environment Setup

```bash
!nvidia-smi          # GPU verification
%ls                  # Checking the files present
```

#### 7.1.2 🔄 Cloning the repository
```bash
!test -d DLA_LLMSANALYSIS && rm -rf DLA_LLMSANALYSIS
!git clone https://github.com/wakaflocka17/DLA_LLMSANALYSIS.git
%cd DLA_LLMSANALYSIS
```

#### 7.1.3 🐍 Creation and activation of the virtual environment
```bash
!pip install virtualenv
!python -m virtualenv venv
!source venv/bin/activate
```

#### 7.1.4 📦 Installing dependencies
```bash
!venv/bin/pip install -r requirements.txt
```

#### 7.1.5 🔐 HuggingFace Login
```python
from huggingface_hub import notebook_login
notebook_login()
```

#### 7.1.6 🧠 Models training and evaluation
##### 🔹 **BERT**
```python
# Training
!venv/bin/python main.py --model_config_key bert_base_uncased --mode train

# Evaluation - pretrained
!venv/bin/python main.py --model_config_key bert_base_uncased --mode eval --eval_type pretrained --output_json_path "results/evaluation/pretrained/bert-base-uncased-imdb.json"

# Evaluation - fine-tuned
!venv/bin/python main.py --model_config_key bert_base_uncased --mode eval --eval_type fine_tuned --output_json_path "results/evaluation/finetuned/bert-base-uncased-imdb.json"
```

##### 🔹 **BART**
```python
# Training
!venv/bin/python main.py --model_config_key bart_base --mode train

# Evaluation - pretrained
!venv/bin/python main.py --model_config_key bart_base --mode eval --eval_type pretrained --output_json_path "results/evaluation/pretrained/bart-base-imdb.json"

# Evaluation - fine-tuned
!venv/bin/python main.py --model_config_key bart_base --mode eval --eval_type fine_tuned --output_json_path "results/evaluation/finetuned/bart-base-imdb.json"
```

##### 🔹 **GPT-Neo**
```python
# Training
!venv/bin/python main.py --model_config_key gpt_neo_2_7b --mode train

# Evaluation - pretrained
!venv/bin/python main.py --model_config_key gpt_neo_2_7b --mode eval --eval_type pretrained --output_json_path "results/evaluation/pretrained/gpt-neo-2.7b-imdb.json"

# Evaluation - fine-tuned
!venv/bin/python main.py --model_config_key gpt_neo_2_7b --mode eval --eval_type fine_tuned --output_json_path "results/evaluation/finetuned/gpt-neo-2.7b-imdb.json"
```

#### 7.1.7 ☁️ Uploading to Hugging Face Hub
```python
!venv/bin/python src/upload_models.py --only bert-base-uncased-imdb
!venv/bin/python src/upload_models.py --only bart-base-imdb
!venv/bin/python src/upload_models.py --only gpt-neo-2.7B-imdb
```

### 7.2 👥 `ensemble_model_evaluation.ipynb`
This notebook performs ensemble **Majority Voting** among the fine-tuned models for the **Sentiment Analysis** task on the IMDb dataset. Following are the steps performed:

#### 7.2.1 ⚙️ Environment Setup

```bash
!nvidia-smi          # GPU verification
%ls                  # Checking the files present
```

#### 7.2.2 🔄 Cloning the repository
```bash
!test -d DLA_LLMSANALYSIS && rm -rf DLA_LLMSANALYSIS
!git clone https://github.com/wakaflocka17/DLA_LLMSANALYSIS.git
%cd DLA_LLMSANALYSIS
```

#### 7.2.3 🐍 Creation and activation of the virtual environment
```bash
!pip install virtualenv
!python -m virtualenv venv
!source venv/bin/activate
```

#### 7.2.4 📦 Installing dependencies
```bash
!venv/bin/pip install -r requirements.txt
```

#### 7.2.5 🔐 HuggingFace Login
```python
from huggingface_hub import notebook_login
notebook_login()
```

#### 7.2.6 ⬇️ Downloading Fine-Tuned models
##### 🔹 **BERT**
```python
# Download
!venv/bin/python src/download_models.py bert_base_uncased
```

##### 🔹 **BART**
```python
# Download
!venv/bin/python src/download_models.py bart_base
```

##### 🔹 **GPT-Neo**
```python
# Download
!venv/bin/python src/download_models.py gpt_neo_2_7b
```

#### 7.2.7 🧠 Ensemble model evaluation
```python
!venv/bin/python src/upload_models.py --only majority-voting-imdb
```

#### 7.2.8 ☁️ Uploading the Ensemble model to Hugging Face Hub
```python
!venv/bin/python src/upload_models.py --only majority-voting-imdb
```

### 7.3 📊 `models_plots_and_results.ipynb`

---

## 8. 📊 Metrics and Outputs <a name="metrics-and-outputs"></a>

### 8.1 📑 Description
Each model evaluation is based on the following metrics:

| Metric      | Description                                      | Formula (Simplified)                            |
|-------------|--------------------------------------------------|-------------------------------------------------|
| Accuracy    | Overall correctness of the model                 | (TP + TN) / (TP + TN + FP + FN)                 |
| Precision   | How many predicted positives are correct         | TP / (TP + FP)                                  |
| Recall      | Ability to detect all true positives             | TP / (TP + FN)                                  |
| F1-Score    | Harmonic mean of precision and recall            | 2 × (Precision × Recall) / (Precision + Recall) |

> Where:
> - **TP** = True Positives  
> - **TN** = True Negatives  
> - **FP** = False Positives  
> - **FN** = False Negatives  

### 8.2 📂 Output Format

The evaluation metrics are saved as `.json` files for each model in the following format:

```json
{
  "accuracy": 0.91,
  "precision": 0.90,
  "recall": 0.91,
  "f1": 0.90
}
```

---

## 9. 🖥️ Hardware and Limitations <a name="hardware-and-limitations"></a>
> [!NOTE]
> 🧪 All training and evaluation were conducted on **Google Colab Pro+** with the following setup:
> - **Runtime environment**: Google Colab Pro+  
> - **GPU**: NVIDIA A100 (40GB VRAM)  
> - **RAM**: High-RAM Instance (≈ 52 GB)  
> - **Backend**: PyTorch with CUDA

> [!WARNING]
> - Training **GPT-Neo** locally (especially on CPU or low-VRAM GPU) may be extremely slow or unstable
> - If using Apple Silicon (M1/M2/M3/M4), consider the **MPS backend** but expect slower inference on large models

---

## 10. 🤝 Contributions <a name="contributions"></a>
Feel free to contribute to the project! 💡  
We welcome improvements, especially in the following areas:
- Adding new Transformer models (e.g. T5, DeBERTa, DistilBERT)
- Improving ensemble strategies (voting, stacking, etc.)
- Suggesting or implementing new evaluation metrics (e.g. calibration, fairness, coverage@k)

### 10.1 📌 How to Contribute <a name="how-to-cite"></a>

1. Fork the repository
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add new evaluation metric"
   ```
4. Push the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a Pull Request on GitHub
> 📬 We’ll review your proposal and get back to you as soon as possible!

---

## 11. 📝 Licenses <a name="licenses"></a>
> [!NOTE]
> **Code**: This repository's source code is licensed under the [Apache License 2.0](./LICENSE). You can read more at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)
>
> **Documentation**: All documentation, including this README, is licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). See the full text in the [LICENSE_DOCS](./LICENSE_DOCS) file.


---

## 12. ❓ How to Cite
```bibtex
@misc{Sentiment-Project,
author       = {Francesco Congiu},
title        = {Sentiment Analysis with Pretrained, Fine-tuned and Ensemble Transformer Models},
howpublished = {\url{https://github.com/wakaflocka17/DLA_LLMSANALYSIS}},
year         = {2025}
}
```