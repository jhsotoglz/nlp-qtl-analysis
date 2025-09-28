# Task 3: Generate word cloud images on phrased corpus
# 1) frequency-based word cloud
# 2) tf-idf–based word cloud

from pathlib import Path
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

# Using phrased data bigrams and trigrams as input
# And the output is two wordcloud images
BASE_DIR  = Path(__file__).resolve().parent.parent
OUT_DIR   = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_TXT = OUT_DIR / "corpus_tokens_phrased.txt"

def load_docs() -> list[str]:
    if not INPUT_TXT.exists():
        raise SystemExit(f"Missing {INPUT_TXT}. Run scripts/phrases.py first.")
    return [
        line.strip()
        for line in INPUT_TXT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

def wordcloud_freq(freqs: dict[str, float], out_path: Path):
    wc = WordCloud(
        width=800, height=800, background_color="white",
        collocations=False,     # prevent auto-joining into bigrams
        prefer_horizontal=0.9,
        random_state=16         # reproducible layout
    )
    wc.generate_from_frequencies(freqs).to_file(str(out_path))

def main():
    docs = load_docs()

    # Frequency-based word cloud
    counts = Counter()
    for line in docs:
        counts.update(line.split())  # tokens are separated by a space
    wordcloud_freq(counts, OUT_DIR / "phrased_freq_wordcloud.png")

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

    wordcloud_freq(tfidf_weights, OUT_DIR / "phrased_tfidf_wordcloud.png")

    print("Saved phrased wordclouds:")
    print(" -", OUT_DIR / "phrased_freq_wordcloud.png")
    print(" -", OUT_DIR / "phrased_tfidf_wordcloud.png")

if __name__ == "__main__":
    main()
