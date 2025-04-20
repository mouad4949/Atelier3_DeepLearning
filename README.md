# Atelier3_DeepLearning
# 🧠 NLP Sequence Models with PyTorch

This repository presents a Natural Language Processing project using **PyTorch**, focusing on two main parts: **Arabic text classification using sequence models (RNN, Bi-RNN, GRU, LSTM)**, and **text generation using fine-tuned GPT-2 transformer model**.

## 📌 Objective

The objective of this project is to get hands-on experience with PyTorch by:
- Building deep learning architectures for sequence modeling in NLP.
- Exploring classical sequence models and transformers.
- Working with Arabic textual data, including data collection, preprocessing, training, and evaluation.

---

## 🧪 Part 1: Text Classification (Arabic)

### 📅 1. Data Collection

- Utilized **web scraping tools** like `Scrapy` and `BeautifulSoup` to collect Arabic texts from various websites on a **specific topic**.
- Each text was assigned a **relevance score** (ranging from 0 to 10) based on manual evaluation.

| Text (Arabic)          | Score |
|------------------------|-------|
| Example Text 1         | 6     |
| Example Text 2         | 7.5   |

### 🧹 2. Data Preprocessing

Implemented a full **Arabic NLP preprocessing pipeline**, including:
- **Tokenization**
- **Stop word removal**
- **Stemming and Lemmatization**
- **Discretization / Normalization**
- **Cleaning and punctuation removal**
- **Vectorization / Padding** (for model readiness)

### 🧠 3. Model Training

Trained and compared multiple **sequence models** using the preprocessed dataset:
- Vanilla **RNN**
- **Bidirectional RNN**
- **GRU**
- **LSTM**

Performed **hyperparameter tuning** (learning rate, batch size, epochs, hidden layers) to optimize performance.

### 📊 4. Evaluation

Evaluated all models using:
- Standard metrics: **Accuracy**, **Loss**, **Precision**, **Recall**, **F1-score**
- Advanced metric: **BLEU score** (adapted to relevance context)
- Comparative analysis between all architectures

---

## 🤖 Part 2: Text Generation using Transformers (GPT-2)

### 🧹 1. Setup and Fine-Tuning

- Installed `pytorch-transformers` (now `transformers` by Hugging Face).
- Loaded the **pre-trained GPT-2 model**.
- Created a **custom dataset** for fine-tuning GPT-2 (can be user-generated data).
- Fine-tuned GPT-2 on this dataset for a more context-aware generation in Arabic.

### ✍️ 2. Text Generation

- Given an initial sentence or prompt, the model was able to **generate full Arabic paragraphs**.
- Sample results showcase GPT-2’s contextual understanding post fine-tuning.

---

## 🧰 Tools & Libraries

- Python
- PyTorch
- Transformers (Hugging Face)
- Scrapy / BeautifulSoup
- NLTK / SpaCy / Farasa (for Arabic preprocessing)
- Matplotlib / Seaborn (for visualization)

---

## 📚 Folder Structure

```
project/
│
├── data/                # Raw and preprocessed datasets
├── models/              # Trained model checkpoints
├── notebooks/           # Training & evaluation notebooks
├── src/                 # Scripts (training, preprocessing, generation)
├── outputs/             # Generated texts, logs, plots
└── README.md            # Project overview
```

---

## 📈 Results

| Model          | Accuracy | F1-score | BLEU Score |
|----------------|----------|----------|------------|
| RNN            | XX%      | XX%      | X.XX       |
| Bi-RNN         | XX%      | XX%      | X.XX       |
| GRU            | XX%      | XX%      | X.XX       |
| LSTM           | XX%      | XX%      | X.XX       |
| GPT-2 (Gen)    | N/A      | N/A      | Human-eval |

---

## ✅ Conclusion

- Gained deep understanding of NLP sequence modeling using PyTorch.
- Compared classical models with modern transformer-based architectures.
- Learned Arabic-specific NLP preprocessing challenges and solutions.
- Successfully built an end-to-end NLP pipeline from data collection to inference.

