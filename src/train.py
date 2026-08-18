"""
train.py
--------
End-to-end pipeline:
1. Load data/reviews.csv
2. Clean text (preprocess.py)
3. TF-IDF vectorization
4. Train Logistic Regression, Linear SVM, Multinomial Naive Bayes
5. Evaluate (accuracy, macro F1, per-class report, confusion matrix)
6. Save plots + metrics + best model + top influential words
"""

import json
import sys
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import clean_text

RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def main():
    print("Loading data...")
    df = pd.read_csv("data/reviews.csv")
    print(f"  {len(df)} reviews loaded. Class balance:\n{df['sentiment'].value_counts()}")

    print("Cleaning text...")
    df["clean_text"] = df["review_text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["sentiment"], test_size=0.2,
        random_state=42, stratify=df["sentiment"]
    )

    print("Building TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=5),
        "Linear SVM": LinearSVC(C=1),
        "Naive Bayes": MultinomialNB(),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring="f1_macro")

        results[name] = {
            "accuracy": round(acc, 4),
            "macro_f1": round(f1, 4),
            "cv_macro_f1_mean": round(cv_scores.mean(), 4),
            "cv_macro_f1_std": round(cv_scores.std(), 4),
            "classification_report": classification_report(y_test, preds, output_dict=True),
        }
        trained_models[name] = model

        # Confusion matrix plot
        cm = confusion_matrix(y_test, preds, labels=model.classes_)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title(f"Confusion Matrix — {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        fname = f"{RESULTS_DIR}/confusion_matrix_{name.replace(' ', '_').lower()}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        print(f"  accuracy={acc:.4f}  macro_f1={f1:.4f}  saved {fname}")

    # Model comparison bar chart
    plt.figure(figsize=(6, 4))
    names = list(results.keys())
    f1s = [results[n]["macro_f1"] for n in names]
    accs = [results[n]["accuracy"] for n in names]
    x = range(len(names))
    plt.bar([i - 0.2 for i in x], accs, width=0.4, label="Accuracy")
    plt.bar([i + 0.2 for i in x], f1s, width=0.4, label="Macro F1")
    plt.xticks(list(x), names, rotation=15)
    plt.ylim(0, 1)
    plt.legend()
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/model_comparison.png", dpi=150)
    plt.close()

    # Best model by macro F1
    best_name = max(results, key=lambda n: results[n]["macro_f1"])
    best_model = trained_models[best_name]
    print(f"\nBest model: {best_name}")

    # Top influential words per class
    feature_names = vectorizer.get_feature_names_out()
    top_words = {}
    if best_name == "Naive Bayes":
        for i, cls in enumerate(best_model.classes_):
            log_probs = best_model.feature_log_prob_[i]
            top_idx = log_probs.argsort()[-15:][::-1]
            top_words[cls] = [feature_names[j] for j in top_idx]
    else:
        coefs = best_model.coef_
        for i, cls in enumerate(best_model.classes_):
            row = coefs[i] if coefs.shape[0] > 1 else coefs[0]
            top_idx = row.argsort()[-15:][::-1]
            top_words[cls] = [feature_names[j] for j in top_idx]

    with open(f"{RESULTS_DIR}/top_words.json", "w") as f:
        json.dump(top_words, f, indent=2)

    with open(f"{RESULTS_DIR}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    joblib.dump(best_model, f"{MODELS_DIR}/best_model_{best_name.replace(' ', '_').lower()}.joblib")
    joblib.dump(vectorizer, f"{MODELS_DIR}/tfidf_vectorizer.joblib")

    # Print summary table
    print("\n=== Summary ===")
    print(f"{'Model':<22}{'Accuracy':<12}{'Macro F1':<12}{'CV F1 (mean±std)'}")
    for n in names:
        r = results[n]
        print(f"{n:<22}{r['accuracy']:<12}{r['macro_f1']:<12}{r['cv_macro_f1_mean']}±{r['cv_macro_f1_std']}")

    print(f"\nTop words for best model ({best_name}):")
    for cls, words in top_words.items():
        print(f"  {cls}: {', '.join(words[:10])}")


if __name__ == "__main__":
    main()
