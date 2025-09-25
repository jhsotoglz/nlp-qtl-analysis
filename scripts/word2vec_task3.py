# Script file for task 3 part 2
# Training word2vec model according to the assignment parameters

from pathlib import Path
import json
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

# Path aligned to preprocessing and phrase script outputs
BASE_DIR = Path(__file__).resolve().parent.parent
PHRASED_JSON = BASE_DIR / "outputs" / "tokenized_abstracts_phrased.json"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_token_lists() -> list[list[str]]:
    """Load phrased (bigram/trigram) tokenized abstracts."""
    if not PHRASED_JSON.exists():
        raise SystemExit(f"Missing {PHRASED_JSON}. Run scripts/phrases.py first.")
    with PHRASED_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)

def train_word2vec(sentences: list[list[str]]) -> Word2Vec:
    """
    Train Word2Vec with the assignment's parameters on the phrased corpus:
    vector_size=100, window=5, min_count=10 (CBOW).
    """
    return Word2Vec(
        sentences=sentences,   # each abstract is a list of tokens (phrases joined by '_')
        vector_size=100,       # embedding dimension
        window=5,              # context window
        min_count=10,          # ignore words/phrases appearing < 10 times
        sg=0,                  # 0 = CBOW
        workers=4,             # threads
        epochs=10,             # passes over corpus
        seed=42                # reproducibility
    )

def top_k_tfidf_terms(docs: list[list[str]], k=10) -> list[str]:
    """Return top-k terms by average TF-IDF across docs (phrased corpus)."""
    docs_as_text = [" ".join(toks) for toks in docs]
    vec = TfidfVectorizer(
        tokenizer=str.split, preprocessor=None, lowercase=False,
        use_idf=True, smooth_idf=True, norm=None
    )
    X = vec.fit_transform(docs_as_text)
    vocab = vec.get_feature_names_out()
    tfidf_avg = (X.sum(axis=0) / X.getnnz(axis=0)).A1
    pairs = sorted(zip(vocab, tfidf_avg), key=lambda x: x[1], reverse=True)
    return [w for w, _ in pairs[:k]]

def main():
    token_lists = load_token_lists()

    # 1) Train Word2Vec on phrased tokens (no files saved)
    model = train_word2vec(token_lists)

    # 2) Top-10 TF-IDF terms from phrased corpus
    top10 = top_k_tfidf_terms(token_lists, k=10)

    # 3) Save neighbors to file
    out_path = OUT_DIR / "task3_neighbors_phrased.txt"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("Top-10 TF-IDF terms (phrased):\n")
        f.write(", ".join(top10) + "\n\n")
        f.write("Nearest neighbors (top 20) for each phrased TF-IDF term:\n")

        for term in top10:
            if term in model.wv.key_to_index:
                sims = model.wv.most_similar(term, topn=20)
                f.write(f"\n[{term}]\n")
                for w, score in sims:
                    f.write(f"  {w:30s}  {score:.4f}\n")
            else:
                f.write(f"\n[{term}] -- not in Word2Vec vocabulary (skipped; appears < 10 times)\n")

    print(f"Results written to {out_path}")

if __name__ == "__main__":
    main()
