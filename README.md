# 🧠 SENTIMENT ANALYSIS 2024/25 - UNICA

<div align="center">
  <img src="https://github.com/user-attachments/assets/c437885d-d304-4714-b4c2-0f47409ed226" alt="sentiment" width="600">
  <p>Project on <b>Binary Sentiment Analysis</b> using <b>Pretrained</b>, <b>Fine-tuned</b> and <b>Ensemble</b> Transformer Models.</p>
</div>

---

> ## 📑 Summary
> 01. [🧑🏻‍🎓 Student](#-student)
> 02. [📌 Description](#-description)
> 03. [📄 Notebooks Overview](#-notebooks-overview)
> 04. [📁 Project Structure](#-project-structure)
> 05. [🔐 Access to Hugging Face Models](#-access-to-hugging-face-models)
> 06. [🚀 Installation](#-installation)
> 07. [🧪 Run: Model Training & Evaluation](#-run-model-training--evaluation)
> 08. [📊 Metrics and Outputs](#-metrics-and-outputs)
> 09. [🖥️ Hardware and Limitations](#hardware-and-limitations)
> 10. [🤝 Contributions](#-contributions)
> 11. [❓ How to Cite](#-how-to-cite)

---

## 🧑🏻‍🎓 Student  

#### Francesco Congiu  
> Student ID: 60/73/65300  
>  
>> E-Mail: f.congiu38@studenti.unica.it  

---

## 📌 Description  
This project investigates the impact of fine-tuning transformer-based models on the **Sentiment Analysis** task using the **IMDb dataset**.  
Three architectures are explored:

1. **Decoder-Only**: GPT-Neo  
2. **Encoder-Only**: BERT  
3. **Encoder-Decoder**: BART  

Additionally, we evaluate the performance of an **ensemble strategy** via **majority voting**.  
Both pretrained and fine-tuned versions are evaluated to compare generalization capabilities.

---

## 📄 Notebooks Overview  

🧾 **Note**: Each notebook is self-contained and was provided for reproducibility.  
Below a quick overview of each file:

| Notebook | Purpose |
|----------|---------|
| `train_models_from_scratch.ipynb` | Fine-tune each model and evaluate them individually |
| `ensemble_model_evaluation.ipynb` | Run ensemble predictions with majority voting |
| `models_plots_and_results.ipynb` | *(Coming soon)* Visual analysis, calibration and fairness plots |

---

## 📁 Project Structure

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

## 🔐 Access to Hugging Face Models

In order to download and use pretrained models from the 🤗 Hugging Face Hub (like `bert-base-uncased`, `gpt-neo-2.7B`, or `bart-base`), you’ll need to authenticate.

### 🪪 How to get your Hugging Face Token

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

## 🚀 Installation
Install requirements for any notebook as needed. For local runs, Python ≥ 3.8 is required.
> [!NOTE]
> For each notebook, you can use a dedicated environment to keep dependencies isolated.

---

## 🧪 Run: Model Training & Evaluation

### 📘 `train_models_from_scratch.ipynb`

### 👥 `ensemble_model_evaluation.ipynb`

### 📊 `models_plots_and_results.ipynb`

---

## 📊 Metrics and Outputs

### 📑 Description
Each model evaluation is based on the following metrics:

| Metric      | Description                                      | Formula (Simplified)                            |
|-------------|--------------------------------------------------|-------------------------------------------------|
| Accuracy    | Overall correctness of the model                 | (TP + TN) / (TP + TN + FP + FN)                 |
| Precision   | How many predicted positives are correct         | TP / (TP + FP)                                  |
| Recall      | Ability to detect all true positives             | TP / (TP + FN)                                  |
| F1-Score    | Harmonic mean of precision and recall            | 2 × (Precision × Recall) / (Precision + Recall) |

Where:
- **TP** = True Positives  
- **TN** = True Negatives  
- **FP** = False Positives  
- **FN** = False Negatives  

### 📂 Output Format

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

## 🖥️ Hardware and Limitations <a name="hardware-and-limitations"></a>
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

## 🤝 Contributions
Feel free to contribute to the project! 💡  
We welcome improvements, especially in the following areas:
- Adding new Transformer models (e.g. T5, DeBERTa, DistilBERT)
- Improving ensemble strategies (voting, stacking, etc.)
- Suggesting or implementing new evaluation metrics (e.g. calibration, fairness, coverage@k)

### 📌 How to Contribute

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

## ❓ How to Cite
```bibtex
@misc{Sentiment-Project,
author       = {Francesco},
title        = {Sentiment Analysis with Pretrained, Fine-tuned and Ensemble Transformer Models},
howpublished = {\url{https://github.com/<your-username>/sentiment-analysis-transformers}},
year         = {2025}
}
```
