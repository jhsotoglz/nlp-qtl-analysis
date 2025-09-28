# Task 1: Generate word cloud images
# 1) frequency-based word cloud
# 2) tf-idf–based word cloud

from pathlib import Path
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

# Using preprocessed data as input
# And the output is two wordclould images
BASE_DIR  = Path(__file__).resolve().parent.parent
INPUT_TXT = BASE_DIR / "outputs" / "corpus_tokens.txt"
OUT_DIR   = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_docs() -> list[str]:
    if not INPUT_TXT.exists():
        raise SystemExit(f"Missing {INPUT_TXT}. Run scripts/preprocess.py first.")
    return [
        line.strip()
        for line in INPUT_TXT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

def wordcloud_freq(freqs: dict[str, float], out_path: Path):
    wc = WordCloud(
        width=800, height=800, background_color="white",
        collocations=False,     # Preventing auto-join words into bigrams
        prefer_horizontal=0.9,
        random_state=16         # to easily reproduce
    )
    wc.generate_from_frequencies(freqs).to_file(str(out_path))

def main():
    docs = load_docs()

    # Frequency-based word cloud
    counts = Counter()
    for line in docs:
        counts.update(line.split())  # tokens are space-separated
    wordcloud_freq(counts, OUT_DIR / "freq_wordcloud_task1.png")

    # TF-IDF–based word cloud
    vec = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        lowercase=False,       # already lowercase
        use_idf=True, smooth_idf=True,
        norm=None              # raw tf-idf weights
    )
    X = vec.fit_transform(docs)
    vocab = vec.get_feature_names_out()

    # Get average TF-IDF for each term across the docs where it appears
    tfidf_avg = (X.sum(axis=0) / X.getnnz(axis=0)).A1
    tfidf_weights = {vocab[i]: float(tfidf_avg[i]) for i in range(len(vocab))}

    wordcloud_freq(tfidf_weights, OUT_DIR / "tfidf_wordcloud_task1.png")

    print("Saved:")
    print(" -", OUT_DIR / "freq_wordcloud_task1.png")
    print(" -", OUT_DIR / "tfidf_wordcloud_task1.png")

if __name__ == "__main__":
    main()