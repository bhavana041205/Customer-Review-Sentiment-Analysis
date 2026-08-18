# Customer Review Sentiment Analysis

Classifies customer product reviews as **positive / negative / neutral** using
classical NLP + machine learning (TF-IDF features with Logistic Regression,
Linear SVM, and Naive Bayes).

## Why this project
Complements regression/tree-based work (XGBoost, Random Forest, SMOTE) with a
text-classification pipeline — a different ML area: NLP preprocessing,
sparse feature engineering, and multi-class text classification.

## Dataset
`data/reviews.csv` — **10,500 reviews** across 15 product categories
(headphones, laptops, kitchen appliances, etc.), generated with
`src/generate_data.py`.

**Note on the data:** this environment has no access to real review platforms
(Amazon/Yelp/Kaggle), so the dataset is synthetically generated from
templates — not scraped. To keep the task realistic (not trivially easy),
the generator deliberately adds:
- mixed-sentiment reviews ("great sound, but the battery died fast")
- ~25% of reviews blending in an opposite-sentiment clause
- ~3% label noise (mislabeled rows, as real human-tagged data has)
- typos, terse one-line reviews, and shared vocabulary across classes

If you swap in a real dataset (Amazon Reviews, Yelp Open Dataset), the
pipeline (`src/preprocess.py`, `src/train.py`) works unchanged — just point
`train.py` at a CSV with `review_text` and `sentiment` columns.

## Pipeline
1. **Preprocessing** (`src/preprocess.py`): lowercasing, URL/punctuation
   removal, stopword removal, Porter stemming.
2. **Feature engineering**: TF-IDF, unigrams + bigrams, top 5,000 features.
3. **Models**: Logistic Regression, Linear SVM, Multinomial Naive Bayes —
   trained on an 80/20 stratified split, validated with 5-fold CV.
4. **Evaluation**: accuracy, macro F1, per-class precision/recall,
   confusion matrices.
5. **Interpretability**: top TF-IDF-weighted words per sentiment class for
   the best model.

## Results

| Model               | Accuracy | Macro F1 | 5-fold CV Macro F1 |
|---------------------|----------|----------|---------------------|
| Logistic Regression | 92.8%    | 0.925    | 0.916 ± 0.004       |
| **Linear SVM**       | **93.0%**| **0.927**| **0.922 ± 0.006**   |
| Naive Bayes          | 90.1%    | 0.896    | 0.892 ± 0.006       |

Linear SVM performed best and was saved as the final model.

**Most influential words (Linear SVM):**
- Negative: worth, disappointing, flimsy, awful, clunky, faulty, terrible
- Neutral: okay, fine, acceptable, decent, reasonable, average
- Positive: impressive, affordable, comfortable, superb, great, responsive

See `results/` for confusion matrix plots per model and a model comparison
chart, and `results/metrics.json` / `results/top_words.json` for full numbers.

## How to run
```bash
pip install -r requirements.txt
python src/generate_data.py   # builds data/reviews.csv
python src/train.py           # trains, evaluates, saves models + plots
```

Outputs land in `results/` (metrics, plots) and `models/` (trained
vectorizer + best model as `.joblib` files, ready to load for inference).

## Project structure
```
customer-review-sentiment-analysis/
├── data/
│   └── reviews.csv
├── src/
│   ├── generate_data.py
│   ├── preprocess.py
│   └── train.py
├── notebooks/
│   └── sentiment_analysis.ipynb
├── results/
│   ├── metrics.json
│   ├── top_words.json
│   ├── model_comparison.png
│   └── confusion_matrix_*.png
├── models/
│   ├── best_model_linear_svm.joblib
│   └── tfidf_vectorizer.joblib
├── requirements.txt
└── README.md
```

## Resume bullet points
- Cleaned and preprocessed 10,500+ customer reviews using NLP techniques
  (tokenization, stopword removal, stemming).
- Engineered TF-IDF (unigram + bigram) features and benchmarked Logistic
  Regression, SVM, and Naive Bayes classifiers with 5-fold cross-validation.
- Achieved 93% accuracy / 0.93 macro F1 with a Linear SVM classifier;
  identified the most influential words per sentiment class for
  interpretability.
