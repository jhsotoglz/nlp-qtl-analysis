from pathlib import Path
import json
from gensim.models.phrases import Phrases, Phraser

BASE_DIR   = Path(__file__).resolve().parent.parent
IN_JSON    = BASE_DIR / "outputs" / "tokenized_abstracts.json"   # from preprocessing
OUT_DIR    = BASE_DIR / "outputs"
OUT_TXT    = OUT_DIR / "tokenized_abstracts_phrased.txt"
OUT_JSON   = OUT_DIR / "tokenized_abstracts_phrased.json"

def load_tokens():
    if not IN_JSON.exists():
        raise SystemExit(f"Missing {IN_JSON}. Run scripts/preprocess.py first.")
    return json.loads(IN_JSON.read_text(encoding="utf-8"))

def save_outputs(token_lists):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # text (one doc per line)
    with OUT_TXT.open("w", encoding="utf-8") as f:
        for toks in token_lists:
            f.write(" ".join(toks) + "\n")
    # json (list[list[str]])
    OUT_JSON.write_text(json.dumps(token_lists), encoding="utf-8")

def main():
    token_lists = load_tokens()

    # Learn bigrams
    # min_count: ignore rare co-occurrences; threshold: higher → fewer phrases (more conservative)
    bigram = Phrases(token_lists, min_count=10, threshold=10.0, delimiter=b"_")
    bigram_phraser = Phraser(bigram)

    # Transform corpus with bigrams
    token_lists_bi = [bigram_phraser[doc] for doc in token_lists]

    # (Optional) Learn trigrams on top of bigrams (often helpful in scientific text)
    trigram = Phrases(token_lists_bi, min_count=10, threshold=10.0, delimiter=b"_")
    trigram_phraser = Phraser(trigram)

    token_lists_tri = [trigram_phraser[doc] for doc in token_lists_bi]

    save_outputs(token_lists_tri)

    print(f"Saved phrased corpus:\n - {OUT_TXT}\n - {OUT_JSON}")

if __name__ == "__main__":
    main()
