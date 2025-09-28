# Phrase extraction for Task 3:
# 1) load preprocessed data
# 2) learn bigrams
# 3) apply bigram model to corpus
# 4) learn trigrams
# 5) apply trigram model to corpus
# 6) save new phrased corpus joined by "_" )

from pathlib import Path
import json
from gensim.models.phrases import Phrases, Phraser

BASE_DIR = Path(__file__).resolve().parent.parent
IN_JSON  = BASE_DIR / "outputs" / "corpus_tokens.json"   # From preprocessing
OUT_DIR  = BASE_DIR / "outputs"
OUT_TXT  = OUT_DIR / "corpus_tokens_phrased.txt"
OUT_JSON = OUT_DIR / "corpus_tokens_phrased.json"

#  I tested different phrase detection parameters to compare:

# Conservative few, but very strong phrases
# MIN_COUNT, THRESHOLD = 10, 10.0

# Balanced, a good middle ground
MIN_COUNT, THRESHOLD = 8, 8.0

# Aggressive, has more phrases but some noise
#MIN_COUNT, THRESHOLD = 5, 5.0

# More aggressive with lots of phrases but lots of noise
# MIN_COUNT, THRESHOLD = 3, 3.0

DELIM = "_"  # This is how the phrases are joined

def load_tokens() -> list[list[str]]:
    if not IN_JSON.exists():
        raise SystemExit(f"Missing {IN_JSON}. Run scripts/preprocess.py first.")
    with IN_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_outputs(token_lists: list[list[str]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_TXT.open("w", encoding="utf-8") as f:
        for toks in token_lists:
            f.write(" ".join(toks) + "\n")
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(token_lists, f)
    print("Saved phrased corpus:")
    print(" -", OUT_TXT)
    print(" -", OUT_JSON)

def main():
    token_lists = load_tokens()

    # Learn bigrams
    bigram = Phrases(token_lists, min_count=MIN_COUNT, threshold=THRESHOLD, delimiter=DELIM)
    bigram_phraser = Phraser(bigram)
    token_lists_bi = [bigram_phraser[doc] for doc in token_lists]

    # Learn trigrams on top of bigrams
    trigram = Phrases(token_lists_bi, min_count=MIN_COUNT, threshold=THRESHOLD, delimiter=DELIM)
    trigram_phraser = Phraser(trigram)
    token_lists_tri = [trigram_phraser[doc] for doc in token_lists_bi]

    save_outputs(token_lists_tri)

if __name__ == "__main__":
    main()
