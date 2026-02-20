# Emotion Detection from Text 💬

**Emotion Detection** is a deep learning project that classifies textual data into six emotion categories using a **GRU-based neural network** built with **TensorFlow** and **Keras**. The model predicts emotions such as happiness, sadness, fear, anger, surprise, and neutral from user-written text.

---

## Dataset

The project uses the [Emotions Dataset](https://www.kaggle.com/nelgiriyewithana/emotions) from Kaggle.

* The dataset contains short textual statements labeled with one of six emotions.

* It is stored in a CSV file `text.csv` with the following columns:

  * `text` — the text content
  * `label` — integer representing the emotion class

* Distribution of classes in the first 10,000 samples:

| Label | Proportion |
| ----- | ---------- |
| 1     | 33.96%     |
| 0     | 29.35%     |
| 3     | 13.67%     |
| 4     | 11.49%     |
| 2     | 7.89%      |
| 5     | 3.64%      |

---

## Data Preprocessing

* Text is **lowercased**, tokenized, lemmatized, and cleaned of stopwords using **NLTK** and **spaCy**.
* Sequences are tokenized and padded to a maximum length for input to the model.
* Class weights are computed to handle class imbalance during training.

---

## Model Architecture

The model is a **Sequential GRU network** with the following layers:

1. **Embedding Layer** – converts words to dense vectors.
2. **GRU Layer (512 units)** – learns temporal patterns in text.
3. **Dense Layers** – with Leaky ReLU activations and regularization.
4. **Dropout** – applied to prevent overfitting.
5. **Output Layer** – 6 units with Softmax activation for emotion classification.

* Loss function: `sparse_categorical_crossentropy`
* Optimizer: `Adam`
* Metrics: `accuracy`

---

## Training

* Dataset split: 80% train, 20% test
* Validation split during training: 15% of training data
* Batch size: 32
* Epochs: 20 with **Early Stopping** (patience = 5)

**Sample Training Results:**

* Initial epochs: model quickly learns patterns from text
* Final accuracy: ~95% on training and ~89% on validation

---

## Features

* Predicts emotions from free-text input.
* Handles imbalanced datasets using class weights.
* Preprocessing pipeline cleans and lemmatizes text for better performance.
* Can be extended to larger datasets or integrated into chatbots, social media analysis, or sentiment monitoring tools.

---

## Requirements

* Python 3.8+
* TensorFlow / Keras
* pandas, numpy
* nltk, spacy
* KaggleHub (for dataset download)

---

## Usage

1. Download the dataset from Kaggle.
2. Preprocess the text using the provided pipeline.
3. Train the GRU model with class weights to handle imbalance.
4. Evaluate the model on the test set.
5. Use the trained model to predict emotions on new text input.

---

## Notes

* Preprocessing is critical for model performance.
* GRU is used for sequence modeling due to its efficiency with textual data.
* Dropout and L2 regularization help reduce overfitting.
* Model performance can be improved with larger datasets or advanced embeddings (e.g., Word2Vec, GloVe, or BERT).


تحب أعملها؟
