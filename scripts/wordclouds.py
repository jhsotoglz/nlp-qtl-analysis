# Script file for Task 1 generate two cloud images
# 1) Word frequency
# 2) use tf-idf

from pathlib import Path
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

# Paths aligned with the preprocessing outputs
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_TXT = BASE_DIR / "outputs" / "corpus_tokens.txt"
OUT_DIR   = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_docs() -> list[str]:
    if not INPUT_TXT.exists():
        raise SystemExit(f"Missing {INPUT_TXT}. Run scripts/preprocess.py first.")
    # each line = one abstract; tokens already lowercased & stopwords removed
    return [line.strip() for line in INPUT_TXT.read_text(encoding="utf-8").splitlines() if line.strip()]

def make_wordcloud_from_freq(freqs: dict[str, float], out_path: Path):
    wc = WordCloud(
        width=800, height=800, background_color="white",
        collocations=False,  # don’t auto-join words into bigrams
        prefer_horizontal=0.9,
        random_state=42      # reproducible layout
    )
    img = wc.generate_from_frequencies(freqs)
    img.to_file(str(out_path))

def main():
    docs = load_docs()

    # Raw frequency word cloud
    counts = Counter()
    for line in docs:
        counts.update(line.split())       # tokens are space-separated
    make_wordcloud_from_freq(counts, OUT_DIR / "wordcloud_frequency_800x800.png")

    # TF-IDF word cloud
    # Already lowercased & tokenized so I am using a simple split tokenizer.
    vec = TfidfVectorizer(
        tokenizer=str.split,   # split on spaces
        preprocessor=None,
        lowercase=False,       # already lowercased
        use_idf=True, smooth_idf=True,
        norm=None              # keep raw tf-idf weights because it seems to be better for wordcloud
    )
    X = vec.fit_transform(docs)
    vocab = vec.get_feature_names_out()

    # Average TF-IDF across the docs where each term appears
    # stable, avoids biasing to very long docs
    tfidf_avg = (X.sum(axis=0) / X.getnnz(axis=0)).A1
    tfidf_weights = {vocab[i]: float(tfidf_avg[i]) for i in range(len(vocab))}

    make_wordcloud_from_freq(tfidf_weights, OUT_DIR / "wordcloud_tfidf_800x800.png")

    print("Saved:")
    print(" -", OUT_DIR / "wordcloud_frequency_800x800.png")
    print(" -", OUT_DIR / "wordcloud_tfidf_800x800.png")

if __name__ == "__main__":
    main()