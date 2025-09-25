# Script file for Task 3 part 1
# Frequency
# TF-IDF

from pathlib import Path
from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR  = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_TXT = OUT_DIR / "tokenized_abstracts_phrased.txt"

def load_docs():
    if not INPUT_TXT.exists():
        raise SystemExit(f"Missing {INPUT_TXT}. Run scripts/phrases.py first.")
    return [line.strip() for line in INPUT_TXT.read_text(encoding="utf-8").splitlines() if line.strip()]

def make_wordcloud_from_freq(freqs, out_path):
    wc = WordCloud(
        width=800, height=800, background_color="white",
        collocations=False, prefer_horizontal=0.9, random_state=42
    )
    wc.generate_from_frequencies(freqs).to_file(str(out_path))

def main():
    docs = load_docs()

    # 1) Frequency word cloud
    counts = Counter()
    for line in docs:
        counts.update(line.split())
    make_wordcloud_from_freq(counts, OUT_DIR / "wordcloud_frequency_phrased_800x800.png")

    # 2) TF-IDF word cloud
    vec = TfidfVectorizer(
        tokenizer=str.split, preprocessor=None, lowercase=False,
        use_idf=True, smooth_idf=True, norm=None
    )
    X = vec.fit_transform(docs)
    vocab = vec.get_feature_names_out()
    tfidf_avg = (X.sum(axis=0) / X.getnnz(axis=0)).A1
    weights = {vocab[i]: float(tfidf_avg[i]) for i in range(len(vocab))}
    make_wordcloud_from_freq(weights, OUT_DIR / "wordcloud_tfidf_phrased_800x800.png")

    print("Saved phrased wordclouds:")
    print(" -", OUT_DIR / "wordcloud_frequency_phrased_800x800.png")
    print(" -", OUT_DIR / "wordcloud_tfidf_phrased_800x800.png")

if __name__ == "__main__":
    main()
