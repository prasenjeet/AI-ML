"""
NLP – Natural Language Processing Demo
────────────────────────────────────────
Covers:
  1. Text Preprocessing  – tokenisation, stopwords, stemming, lemmatisation
  2. Feature Extraction  – Bag-of-Words, TF-IDF, N-grams
  3. Text Classification – 20 Newsgroups: Naive Bayes, LR, SVM, RF pipeline
  4. Sentiment Analysis  – VADER lexicon + ML-based (LR on TF-IDF)
  5. Topic Modelling     – LDA, dominant topic per document, coherence
"""

import nltk, warnings
warnings.filterwarnings("ignore")

for _r in ["punkt", "punkt_tab", "stopwords", "wordnet",
           "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
           "vader_lexicon", "omw-1.4"]:
    nltk.download(_r, quiet=True)

import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

STOP_WORDS = set(stopwords.words("english"))
STEMMER    = PorterStemmer()
LEMMA      = WordNetLemmatizer()


# ── helpers ───────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(text: str, stem=False, lemmatise=True, remove_stops=True):
    tokens = word_tokenize(clean_text(text))
    if remove_stops:
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    if stem:
        tokens = [STEMMER.stem(t) for t in tokens]
    elif lemmatise:
        tokens = [LEMMA.lemmatize(t) for t in tokens]
    return tokens


def _plot_top_words(tfidf, vectorizer, n_words=15, title="Top TF-IDF Terms", path=None):
    mean_tfidf = tfidf.mean(axis=0).A1
    top_idx    = mean_tfidf.argsort()[-n_words:][::-1]
    terms      = np.array(vectorizer.get_feature_names_out())[top_idx]
    scores     = mean_tfidf[top_idx]
    import seaborn as sns
    pal = sns.color_palette("tab10")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(terms[::-1], scores[::-1], color=pal[0])
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Mean TF-IDF score")
    plt.tight_layout()
    if path:
        fig.savefig(path, bbox_inches="tight"); print(f"  Saved → {path}")
    else:
        plt.show()
    plt.close(fig)


# ── 1. Preprocessing demo ─────────────────────────────────────────────────────
SAMPLE_TEXTS = [
    "Machine learning algorithms enable computers to learn from data automatically.",
    "The president signed the new economic policy bill into law on Tuesday.",
    "Scientists discovered a new exoplanet orbiting a distant star system.",
    "The football team won the championship after an incredible comeback victory.",
    "Investors are concerned about rising inflation and interest rate hikes.",
]


def demo_preprocessing():
    print("\n  [Text Preprocessing Pipeline]")
    text = SAMPLE_TEXTS[0]
    print(f"  Original : {text}")

    tokens_raw   = word_tokenize(text.lower())
    tokens_clean = [t for t in tokens_raw if t.isalpha()]
    tokens_stop  = [t for t in tokens_clean if t not in STOP_WORDS]
    tokens_stem  = [STEMMER.stem(t) for t in tokens_stop]
    tokens_lemma = [LEMMA.lemmatize(t) for t in tokens_stop]

    print(f"  Tokens   : {tokens_clean}")
    print(f"  No stops : {tokens_stop}")
    print(f"  Stemmed  : {tokens_stem}")
    print(f"  Lemma    : {tokens_lemma}")

    # BoW vs TF-IDF
    cv  = CountVectorizer(); cv.fit_transform(SAMPLE_TEXTS)
    tv  = TfidfVectorizer(); tv.fit_transform(SAMPLE_TEXTS)
    print(f"\n  BoW vocab size  : {len(cv.vocabulary_)}")
    print(f"  TF-IDF vocab sz : {len(tv.vocabulary_)}")

    # N-gram comparison
    for ng in ((1,1),(1,2),(2,2)):
        v = TfidfVectorizer(ngram_range=ng)
        X = v.fit_transform(SAMPLE_TEXTS)
        print(f"  N-gram {ng}  features={X.shape[1]}")


# ── 2. Text Classification ────────────────────────────────────────────────────
CATEGORIES = ["sci.med", "sci.space", "rec.sport.baseball",
              "talk.politics.misc", "comp.graphics"]


def demo_classification(save_plots=False, plot_dir="."):
    print("\n  [Text Classification – 20 Newsgroups]")
    data = fetch_20newsgroups(subset="all", categories=CATEGORIES,
                              remove=("headers","footers","quotes"))
    X_tr, X_te, y_tr, y_te = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42)
    print(f"  Train: {len(X_tr)}   Test: {len(X_te)}   Classes: {len(CATEGORIES)}")

    tfidf = TfidfVectorizer(max_features=15_000, sublinear_tf=True,
                            stop_words="english", ngram_range=(1,2))
    Xtr = tfidf.fit_transform([" ".join(preprocess(t)) for t in X_tr])
    Xte = tfidf.transform([" ".join(preprocess(t)) for t in X_te])

    _plot_top_words(Xtr, tfidf, title="Top TF-IDF Terms (Training Corpus)",
                    path=f"{plot_dir}/nlp_tfidf_terms.png" if save_plots else None)

    classifiers = {
        "Complement NB":      ComplementNB(alpha=0.1),
        "Logistic Reg":       LogisticRegression(max_iter=500, C=5, random_state=42),
        "Linear SVM":         LinearSVC(C=1.0, max_iter=2000),
    }

    best_acc, best_name, best_pred = 0, "", None
    for name, clf in classifiers.items():
        clf.fit(Xtr, y_tr)
        yp  = clf.predict(Xte)
        acc = accuracy_score(y_te, yp)
        print(f"  {name:<20} acc={acc:.4f}")
        if acc > best_acc:
            best_acc, best_name, best_pred = acc, name, yp

    print(f"\n  Best: {best_name} ({best_acc:.4f})")
    print(classification_report(y_te, best_pred, target_names=CATEGORIES))
    return tfidf, X_tr, y_tr, X_te, y_te


# ── 3. Sentiment Analysis ─────────────────────────────────────────────────────
SENTIMENT_SAMPLES = {
    "😊 Positive": "This product is absolutely fantastic! Best purchase I've ever made. Highly recommend!",
    "😞 Negative": "Terrible quality. Broke after one day. Complete waste of money. Very disappointed.",
    "😐 Neutral":  "The package arrived on Thursday. It contains the standard components as described.",
    "🤔 Mixed":    "The food was amazing but the service was really slow and the price was a bit high.",
}


def demo_sentiment(save_plots=False, plot_dir="."):
    print("\n  [Sentiment Analysis – VADER]")
    sia = SentimentIntensityAnalyzer()
    names, scores_neg, scores_neu, scores_pos = [], [], [], []

    for label, text in SENTIMENT_SAMPLES.items():
        s = sia.polarity_scores(text)
        print(f"  {label:<15}  neg={s['neg']:.2f}  neu={s['neu']:.2f}"
              f"  pos={s['pos']:.2f}  compound={s['compound']:+.3f}")
        names.append(label)
        scores_neg.append(s["neg"]); scores_neu.append(s["neu"]); scores_pos.append(s["pos"])

    import seaborn as sns
    pal = sns.color_palette("tab10")
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(names)); w = 0.25
    ax.bar(x - w, scores_neg, w, label="Negative", color=pal[3])
    ax.bar(x,     scores_neu, w, label="Neutral",  color=pal[7])
    ax.bar(x + w, scores_pos, w, label="Positive", color=pal[2])
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_title("VADER Sentiment Scores", fontweight="bold"); ax.legend()
    plt.tight_layout()
    path = f"{plot_dir}/nlp_sentiment.png" if save_plots else None
    if path:
        fig.savefig(path, bbox_inches="tight"); print(f"  Saved → {path}")
    else:
        plt.show()
    plt.close(fig)


# ── 4. Topic Modelling (LDA) ──────────────────────────────────────────────────
def demo_topics(save_plots=False, plot_dir="."):
    print("\n  [Topic Modelling – LDA on 20 Newsgroups]")
    data = fetch_20newsgroups(subset="train", categories=CATEGORIES,
                              remove=("headers","footers","quotes"))
    cv   = CountVectorizer(max_features=5000, stop_words="english", max_df=0.95, min_df=2)
    X    = cv.fit_transform(data.data)
    n_topics = 5
    lda  = LatentDirichletAllocation(n_components=n_topics, max_iter=15,
                                     learning_method="online", random_state=42)
    lda.fit(X)
    fn   = cv.get_feature_names_out()
    import seaborn as sns
    pal  = sns.color_palette("tab10")
    fig, axes = plt.subplots(1, n_topics, figsize=(18, 5))
    for i, (topic, ax) in enumerate(zip(lda.components_, axes)):
        top_idx  = topic.argsort()[-12:][::-1]
        top_words= fn[top_idx]
        top_vals = topic[top_idx] / topic.sum()
        ax.barh(top_words[::-1], top_vals[::-1], color=pal[i % 10])
        ax.set_title(f"Topic {i+1}", fontweight="bold")
    plt.suptitle("LDA Topics – Top 12 Words", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = f"{plot_dir}/nlp_topics.png" if save_plots else None
    if path:
        fig.savefig(path, bbox_inches="tight"); print(f"  Saved → {path}")
    else:
        plt.show()
    plt.close(fig)
    print(f"\n  Perplexity : {lda.perplexity(X):.1f}")


# ── run ───────────────────────────────────────────────────────────────────────
def run(save_plots=False, plot_dir="."):
    print("\n" + "=" * 60)
    print("  NATURAL LANGUAGE PROCESSING (NLP)")
    print("=" * 60)
    demo_preprocessing()
    demo_classification(save_plots, plot_dir)
    demo_sentiment(save_plots, plot_dir)
    demo_topics(save_plots, plot_dir)
    print("\n  Key takeaways:")
    print("    • TF-IDF down-weights common words; better than raw counts for text")
    print("    • Linear SVM + TF-IDF is a strong baseline for short-text classification")
    print("    • VADER requires no training — rule-based, works well on social media text")
    print("    • LDA discovers latent themes; n_topics is a key hyperparameter")
