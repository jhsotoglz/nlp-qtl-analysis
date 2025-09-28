# Task 2: Training Word2Vec model and analyze top TF-IDF terms
# 1) load preprocessed data
# 2) train Word2Vec with CBOW, vector_size=100, window=5, min_count=10
# 3) get the top 10 TF-IDF terms from the corpus
# 4) find 20 most similar words in the Word2Vec space
# 5) save results to outputs/task2_word2vec.txt

from pathlib import Path
import json
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

# Project paths aligned with preprocessing
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
    return Word2Vec(
        sentences=sentences,   # each abstract is a list of words
        vector_size=100,       # length of word vector
        window=5,              # context window
        min_count=10,          # skips words that show up less than 10 times
        sg=0,                  # 0 = CBOW
        workers=4,             # cpu threads in parallel
        epochs=10,             # times of passes over the corpus
        seed=16                # to easy reproduce
    )

def top_n_tfidf_terms(docs: list[list[str]], n=10) -> list[str]:
    docs_as_text = [" ".join(toks) for toks in docs]
    vec = TfidfVectorizer(
        tokenizer=str.split, preprocessor=None, lowercase=False,
        use_idf=True, smooth_idf=True, norm=None
    )
    X = vec.fit_transform(docs_as_text)
    vocab = vec.get_feature_names_out()
    # average TF-IDF where each term appears
    tfidf_avg = (X.sum(axis=0) / X.getnnz(axis=0)).A1
    pairs = sorted(zip(vocab, tfidf_avg), key=lambda x: x[1], reverse=True)
    return [w for w, _ in pairs[:n]]

def main():
    token_lists = load_token_lists()

    # Train Word2Vec in-memory
    model = train_word2vec(token_lists)

    # Get the top-10 TF-IDF terms
    top10 = top_n_tfidf_terms(token_lists, n=10)

    # New outputt file to save results
    out_path = OUT_DIR / "task2_word2vec.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Top-10 TF-IDF terms:\n")
        f.write(", ".join(top10) + "\n\n")

        f.write("Top 20 of nearest neighbors for each TF-IDF term:\n")
        for term in top10:
            if term in model.wv.key_to_index:
                sims = model.wv.most_similar(term, topn=20)
                f.write(f"\n------ {term} ------\n")
                for w, score in sims:
                    f.write(f"  {w:20s}  {score:.4f}\n")
            else:
                f.write(f"\n[{term}] -- Skip words that appear less than 10 times. \n")

    print(f"\nResults written to: {out_path}")

if __name__ == "__main__":
    main()
