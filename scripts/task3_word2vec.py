# Task 3: Training Word2Vec model on phrased corpus and analyze top TF-IDF terms
# 1) load phrased tokenized abstracts
# 2) train Word2Vec with CBOW, vector_size=100, window=5, min_count=10
# 3) get the top 10 TF-IDF terms from the phrased corpus
# 4) find 20 most similar words/phrases in the Word2Vec space
# 5) compare phrases with trait dictionary by exact matching (raw + underscore-normalized)
# 6) save results to outputs/task3_word2vec_phrased.txt

from pathlib import Path
import json
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

# Project paths aligned to preprocessing and phrases outputs
BASE_DIR = Path(__file__).resolve().parent.parent
PHRASED_JSON = BASE_DIR / "outputs" / "corpus_tokens_phrased.json"

# Candidate locations for Trait_dictionary.txt (local first, then NOVA)
TRAIT_DICT_CANDIDATES = [
    BASE_DIR / "datasets" / "Trait_dictionary.txt",
    Path("/work/classtmp/NLP/project_data/Trait_dictionary.txt"),
]

OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_token_lists() -> list[list[str]]:
    if not PHRASED_JSON.exists():
        raise SystemExit(f"Missing {PHRASED_JSON}. Run scripts/phrases.py first.")
    with PHRASED_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)

def train_word2vec(sentences: list[list[str]]) -> Word2Vec:
    return Word2Vec(
        sentences=sentences,   # each abstract is a list of tokens/phrases (underscored)
        vector_size=100,       # length of each word or phrase vector
        window=5,              # how many tokens left and right to look at
        min_count=10,          # ignore tokens that appear fewer than 10 times in the corpus
        sg=0,                  # 0 = CBOW (predict center from context), 1 = Skip-gram
        workers=4,             # number of CPU threads to use in parallel
        epochs=10,             # full passes over the training corpus
        seed=16                # random seed for reproducibility (match Task 2 style)
    )

def top_n_tfidf_terms(docs: list[list[str]], n=10) -> list[str]:
    docs_as_text = [" ".join(toks) for toks in docs]
    vec = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        lowercase=False,
        use_idf=True,
        smooth_idf=True,
        norm=None,
        token_pattern=None,   # silence warning since we provide a tokenizer
    )
    X = vec.fit_transform(docs_as_text)
    vocab = vec.get_feature_names_out()
    # average TF-IDF over docs where each term appears, avoids bias toward long docs
    tfidf_avg = (X.sum(axis=0) / X.getnnz(axis=0)).A1
    pairs = sorted(zip(vocab, tfidf_avg), key=lambda x: x[1], reverse=True)
    return [w for w, _ in pairs[:n]]

def resolve_trait_dict() -> Path:
    for p in TRAIT_DICT_CANDIDATES:
        if p.exists():
            return p
    raise SystemExit(
        "Trait_dictionary.txt not found. Looked in:\n  " +
        "\n  ".join(str(p) for p in TRAIT_DICT_CANDIDATES)
    )

def load_trait_dict(path: Path) -> set[str]:
    """Load trait dictionary terms (lowercased)."""
    with path.open("r", encoding="utf-8") as f:
        return set(line.strip().lower() for line in f if line.strip())

def main():
    token_lists = load_token_lists()

    # Train Word2Vec in-memory on phrased tokens
    model = train_word2vec(token_lists)

    # Get the top-10 TF-IDF terms from the phrased corpus
    top10 = top_n_tfidf_terms(token_lists, n=10)

    # Save neighbors to file
    out_path = OUT_DIR / "task3_word2vec_phrased.txt"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("Top-10 TF-IDF terms (phrased):\n")
        f.write(", ".join(top10) + "\n\n")

        f.write("Top 20 nearest neighbors for each phrased TF-IDF term:\n")
        for term in top10:
            if term in model.wv.key_to_index:
                sims = model.wv.most_similar(term, topn=20)
                f.write(f"\n------ {term} ------\n")
                for w, score in sims:
                    f.write(f"  {w:30s}  {score:.4f}\n")
            else:
                f.write(f"\n[{term}] -- Skip words that appear less than 10 times.\n")

        # Trait dictionary matching
        trait_dict_path = resolve_trait_dict()
        trait_terms_raw = load_trait_dict(trait_dict_path)

        # Flatten all tokens from phrased corpus
        all_phrases = {tok.lower() for doc in token_lists for tok in doc}

        # 1) Raw exact matches
        raw_matches = all_phrases.intersection(trait_terms_raw)

        # 2) Conver exact matches to be joined by "_":
        trait_terms_underscored = {t.replace(" ", "_") for t in trait_terms_raw}
        underscored_matches = all_phrases.intersection(trait_terms_underscored)

        f.write("\n\nTrait Dictionary Matching (Exact):\n")
        f.write(f"Raw exact matches (as-is): {len(raw_matches)} out of {len(trait_terms_raw)} traits.\n")
        if raw_matches:
            f.write("Raw match examples:\n")
            for m in sorted(list(raw_matches))[:20]:
                f.write(f"  {m}\n")

        f.write(f"\nUnderscore-normalized matches: {len(underscored_matches)} out of {len(trait_terms_underscored)} traits.\n")
        if underscored_matches:
            f.write("Underscore-normalized examples:\n")
            for m in sorted(list(underscored_matches))[:20]:
                f.write(f"  {m}\n")

    print(f"\nResults written to: {out_path}")
    print(f"Trait dictionary used: {trait_dict_path}")
    print(f"Raw matches: {len(raw_matches)} | Underscore-normalized matches: {len(underscored_matches)}")

if __name__ == "__main__":
    main()
