# Task 2
# Training a word2vec model

from pathlib import Path
import json
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
TOK_JSON = BASE_DIR / "outputs" / "corpus_tokens.json"  # from preprocessing
OUT_DIR  = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_token_lists() -> list[list[str]]:
    if not TOK_JSON.exists():
        raise SystemExit(f"Missing {TOK_JSON}. Run scripts/preprocess.py first.")
    with open(TOK_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def train_word2vec(sentences: list[list[str]]) -> Word2Vec:
    """
    Train Word2Vec with the assignment's required parameters:
    vector_size=100, window=5, min_count=10, CBOW.
    """
    return Word2Vec(
        sentences=sentences,   # each abstract is a list of words
        vector_size=100,       # embedding dimension
        window=5,              # context window size
        min_count=10,          # ignore words appearing < 10 times
        sg=0,                  # 0 = CBOW (1 = Skip-gram)
        workers=4,             # number of parallel worker threads
        epochs=10,             # training epochs
        seed=42                # reproducibility
    )

def top_k_tfidf_terms(docs: list[list[str]], k=10) -> list[str]:
    """Return the top-k words ranked by average TF-IDF weight."""
    docs_as_text = [" ".join(toks) for toks in docs]
    vec = TfidfVectorizer(
        tokenizer=str.split, preprocessor=None, lowercase=False,
        use_idf=True, smooth_idf=True, norm=None
    )
    X = vec.fit_transform(docs_as_text)
    vocab = vec.get_feature_names_out()
    # average TF-IDF across docs where each term appears
    tfidf_avg = (X.sum(axis=0) / X.getnnz(axis=0)).A1
    pairs = sorted(zip(vocab, tfidf_avg), key=lambda x: x[1], reverse=True)
    return [w for w, _ in pairs[:k]]

def main():
    token_lists = load_token_lists()

    # 1) Train Word2Vec in-memory
    model = train_word2vec(token_lists)

    # 2) Compute top-10 TF-IDF terms
    top10 = top_k_tfidf_terms(token_lists, k=10)

    # 3) Open file to save results
    out_path = OUT_DIR / "task2_neighbors.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Top-10 TF-IDF terms (candidate anchors):\n")
        f.write(", ".join(top10) + "\n\n")

        f.write("Nearest neighbors (top 20) for each TF-IDF term:\n")
        for term in top10:
            if term in model.wv.key_to_index:
                sims = model.wv.most_similar(term, topn=20)
                f.write(f"\n[{term}]\n")
                for w, score in sims:
                    f.write(f"  {w:20s}  {score:.4f}\n")
            else:
                f.write(f"\n[{term}] -- not in Word2Vec vocabulary (skipped; appears < 10 times)\n")

    print(f"\nResults written to {out_path}")

if __name__ == "__main__":
    main()
