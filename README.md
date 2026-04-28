# Fake News Detection

A two-part project that classifies news articles as **Fake** or **Real** using both classical machine learning and modern transformer-based deep learning. Built on the [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) by Clément Bisaillon.

The repo contains two self-contained notebooks that approach the same problem from different angles, so you can compare a fast TF-IDF baseline against a fine-tuned DistilBERT model.

## Notebook 1 — Classical Machine Learning

`fake_news_detection_ml.ipynb`

A lightweight TF-IDF pipeline that trains and compares four classifiers, then uses the best one for prediction.

**Pipeline**
- Loads the dataset directly inside the runtime via `kagglehub` (no manual upload).
- Combines the article title and body into a single `content` field.
- Cleans text by lowercasing and stripping URLs, HTML tags, brackets, and non-alphabetic characters.
- Vectorizes with TF-IDF using unigrams and bigrams (`max_features=50000`, `min_df=2`, `max_df=0.8`, sublinear TF).
- Trains four models and picks the most accurate one:
  - Logistic Regression
  - Linear SVM (`LinearSVC`)
  - Passive Aggressive Classifier
  - Multinomial Naive Bayes
- Evaluates with accuracy, classification report, confusion matrix, and 5-fold cross-validation.
- Provides a `predict_news()` helper and an interactive loop for pasting your own articles.

## Notebook 2 — DistilBERT Deep Learning

`fake_news_detection_distilbert.ipynb`

A fine-tuned `distilbert-base-uncased` classifier built with Hugging Face Transformers and the `Trainer` API.

**Pipeline**
- Loads the same dataset via `kagglehub`.
- Applies light cleaning only (URLs and HTML), preserving sentence structure for the transformer.
- Optional balanced sampling (default `SAMPLE_SIZE = 12000`) for faster experimentation, with a `USE_FULL_DATASET` flag for the full run.
- 70 / 15 / 15 train / validation / test split, stratified by label.
- Tokenizes with the DistilBERT tokenizer (`MAX_LENGTH=256`).
- Fine-tunes for 2 epochs with `learning_rate=2e-5`, `weight_decay=0.01`, early stopping on validation F1.
- Evaluates with accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and ROC curve.
- Adds a **confidence threshold** (default `0.70`) — predictions below the threshold are flagged as *Uncertain / Needs Verification* instead of being forced into Fake/True.
- Includes an **attention-based explanation** function that surfaces the tokens the `[CLS]` representation attended to most in the final layer.
- Includes example tests and a manual testing loop.

## Dataset

Both notebooks pull from the same source:

- **Fake and Real News Dataset** — `clmentbisaillon/fake-and-real-news-dataset` on Kaggle
- Two CSVs: `Fake.csv` and `True.csv`
- Articles are labelled `0 = Fake News`, `1 = True News`
- Title and body are concatenated into a single text field
