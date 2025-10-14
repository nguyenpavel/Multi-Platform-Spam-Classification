# Multi-Platform Spam Classification Tool

A comparative study and prototype tool for spam detection across email, SMS, and social media. Three model families are implemented and evaluated under domain-shift scenarios:

* **TF-IDF + Naive Bayes**
* **FastText embeddings + MLP**
* **DistilBERT (transformer), fine-tuned**

The project assesses how models trained on one platform generalise to others, and provides a minimal web proof-of-concept.

---

## Overview

Spam remains a major vector for phishing and malware. As generative AI raises the quality of spam content, robust, adaptable detectors are required. This project compares traditional, neural, and transformer-based approaches, with a specific focus on **cross-platform robustness**.

**Objectives**

* Build strong baselines for email spam detection.
* Stress-test models under **domain shift** (train on emails, test on SMS + social media).
* Quantify trade-offs among accuracy, compute cost, and generalisation.
* Ship an MVP website for interactive inference.

**Key findings**

* **Naive Bayes** is a strong, lightweight baseline (≈7s to train) with accuracy ≈0.96–0.99 on in-domain data.
* **Fine-tuned DistilBERT** achieves the best in-domain performance (up to **0.992 accuracy**) and benefits from TF-IDF filtering and de-duplication to reduce training time.
* All models **drop sharply** when trained on emails and tested on SMS/social media; precision/recall trade-offs emerge, indicating limited cross-platform transfer without additional data and adaptation.
* Random, simple models are comparatively stable; complex models capture more context but over-generalise when domains diverge.

---

## Datasets

* **Email dataset (Enron-like)** — 33,716 rows; Spam 50.93%, Not spam 49.07%.
* **SMS dataset (UCI)** — 5,572 rows; Spam 13.41%, Not spam 86.59%.
* **YouTube comments (HF)** — 1,645 rows; Spam 46.44%, Not spam 53.56%.

Potential biases: class imbalance and topic skew (e.g., frequent “Eminem” mentions in YouTube set) may affect generalisability.

---

## Methods

### Model 1 — TF-IDF + Naive Bayes

* **Preprocessing**: regex cleaning (e.g., phone numbers → `phonenumber`), lowercasing, lemmatisation (spaCy), removal of 371 missing values.
* **Features**: TF-IDF over tokenised text.
* **Classifier**: Multinomial Naive Bayes.

### Model 2 — FastText + MLP

* **Preprocessing**: as above, plus contraction expansion.
* **Embeddings**: FastText subword vectors (robust to misspellings/OOV).
* **Classifier**: MLP (ReLU hidden layer, sigmoid output), SGD (lr=0.01), ~150 epochs with early stopping in final setting.

### Model 3 — DistilBERT (Transformer)

* **Tokenisation**: DistilBertTokenizerFast (uncased), special tokens `[CLS]`, `[SEP]`.
* **Normalisation**: `[EMAIL]`, `[URL]`; HTML stripping; case-folding; duplicates removed via cosine similarity > 0.8.
* **Training**: fine-tuning with and without TF-IDF filtering (retain 5–95% terms) and near-duplicate removal to shorten training while preserving performance.

---

## Evaluation Protocols

1. **Method 1 (Merged)** — Train/validate/test on a stratified merge of all three datasets.
2. **Method 2 (Email-only, control)** — Train/validate/test on emails only.
3. **Method 3 (Domain shift)** — Train on emails; test on **SMS + YouTube**.

Metrics: **Accuracy, Precision, Recall, F1**. Training time reported for comparability.

---

## Results

### Method 1 — Train/Val/Test on merged datasets

| Model                                   |    Acc    |     F1    |    Prec   |   Recall  |      Train Time |
| --------------------------------------- | :-------: | :-------: | :-------: | :-------: | --------------: |
| Naive Bayes                             |   0.958   |   0.958   |   0.958   |   0.958   |    0:00:07 (7s) |
| MLP                                     |   0.963   |   0.957   |   0.959   |   0.956   | 0:18:57 (1137s) |
| DistilBERT (pretrained, no FT)          |   0.481   |   0.553   |   0.453   |   0.709   |         0:00:00 |
| DistilBERT fine-tuned (2 epochs)        |   0.988   |   0.987   |   0.988   |   0.986   | 0:44:46 (2686s) |
| DistilBERT fine-tuned (ES @ 6 ep.)      |   0.987   |   0.986   |   0.989   |   0.984   | 1:46:06 (6366s) |
| DistilBERT FT + TF-IDF filter (2 ep.)   | **0.992** | **0.991** | **0.991** | **0.989** | 0:39:01 (2341s) |
| DistilBERT FT + de-dup (2 ep.)          |   0.980   |   0.975   |   0.982   |   0.968   | 0:28:02 (1682s) |
| DistilBERT FT + TF-IDF + de-dup (2 ep.) |   0.984   |   0.980   |   0.990   |   0.972   | 0:25:33 (1533s) |

### Method 2 — Email-only (control)

| Model                            |  Acc  |   F1  |  Prec | Recall |      Train Time |
| -------------------------------- | :---: | :---: | :---: | :----: | --------------: |
| Naive Bayes                      | 0.985 | 0.985 | 0.985 |  0.985 |    0:00:07 (7s) |
| MLP                              | 0.978 | 0.979 | 0.973 |  0.985 |  0:07:18 (438s) |
| DistilBERT fine-tuned (2 epochs) | 0.993 | 0.993 | 0.991 |  0.994 | 0:34:32 (2072s) |

### Method 3 — Train on emails, test on SMS + YouTube (domain shift)

| Model                            |  Acc  |   F1  |  Prec | Recall |      Train Time |
| -------------------------------- | :---: | :---: | :---: | :----: | --------------: |
| Naive Bayes                      | 0.400 | 0.534 | 0.537 |  0.543 |    0:00:07 (7s) |
| MLP                              | 0.399 | 0.364 | 0.234 |  0.822 |  0:07:18 (438s) |
| DistilBERT fine-tuned (2 epochs) | 0.496 | 0.431 | 0.282 |  0.912 | 0:34:32 (2072s) |

**Interpretation**

* In-domain, DistilBERT dominates; TF-IDF filtering and de-duplication reduce training time with minor performance changes.
* Under domain shift, all models degrade; DistilBERT and MLP tend toward **high recall / low precision**, over-flagging spam in unfamiliar domains.
* Naive Bayes remains comparatively stable but also experiences a significant drop.

---

## Error Analysis (high level)

* **LIME** on DistilBERT reveals misclassifications driven by common words with low semantic salience, indicating uncertainty on out-of-domain text.
* **Confusion matrices** show a bias toward predicting “spam” on SMS/YouTube despite those sets being majority “not spam”, suggesting feature drift across platforms.
* As model complexity increases, false positives decrease (NB → MLP → DistilBERT), but precision remains low under domain shift.

---

## Limitations

* Training data do not fully represent the breadth of real-world spam across platforms; topic skew and class imbalance persist.
* Cross-platform transfer without adaptation is weak; additional data and domain adaptation are required.
* Transformers demand substantial compute; marginal in-domain gains may not justify costs for simple deployments.

---

## Future Work

* Expand and rebalance datasets; add platform-specific corpora and active-learning loops.
* Domain adaptation: adversarial training, feature alignment, or lightweight adapters (LoRA/IA3).
* Hard-negative mining to improve precision on SMS/social media.
* Ensemble methods combining NB stability with transformer recall.
* Browser extension to hide suspected spam and collect user-approved, anonymised feedback.

---

## Getting Started

### Environment

Python 3.10+. Suggested libraries: `pandas`, `numpy`, `scikit-learn`, `spacy`, `fasttext`, `torch`, `transformers`, `datasets`, `matplotlib`, `scipy`.

Conda:

bash
conda create -n spam-nlp python=3.10
conda activate spam-nlp
pip install -r env/requirements.txt


Pip:

bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r env/requirements.txt


FastText wheels and PyTorch may require platform-specific installs. If transformer training fails, install a CPU or CUDA-specific torch wheel first, then `pip install transformers`.

### Reproducibility

* Use fixed random seeds for Python/NumPy/PyTorch.
* Log library versions and hardware (CPU/GPU).
* Ensure consistent text normalisation across datasets (e.g., email/url masking).
