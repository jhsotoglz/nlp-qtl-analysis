import json
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Paths to local datasets
BASE_DIR = Path(__file__).resolve().parent.parent
QTL_JSON = BASE_DIR / "datasets" / "QTL_text.json"
TRAIT_DICT = BASE_DIR / "datasets" / "Trait_dictionary.txt"
OUTPUT_DIR = BASE_DIR / "outputs"

# Paths to datasets on NOVA
# QTL_JSON = Path("/work/classtmp/NLP/project_data/QTL_text.json")
# TRAIT_DICT = Path("/work/classtmp/NLP/project_data/Trait_dictionary.txt")

def load_qtl_json(path):
    """ Load the QTL_text.json dataset. """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)     # This will return a List[dict] with Keys: PMID, Journal, Title, Abstract, and Category fields
    
def preprocess_text(text: str, stop_word_set: set[str]) -> list[str]:
    """ Pre-processing requirements: Split sentences, tokenize, lowercase, remove stop words and unwanted characters. """
    tokens = []
    for sentence in sent_tokenize(text):
        for token in word_tokenize(sentence):
            word = token.lower()
            if word.isalpha() and word not in stop_word_set:
                tokens.append(word)
    return tokens

def main():
    # Creates the outputs/ and prepares the NLTK resources
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    stop_word_set = set(stopwords.words("english"))

    # Pre-processing requirement of only getting abstracts with Category == "1"
    data = load_qtl_json(QTL_JSON)
    abstracts = [d.get("Abstract", "") for d in data if str(d.get("Category", "")).strip() == "1"]

    docs_tokens = [preprocess_text(abs_text, stop_word_set) for abs_text in abstracts]

    with open(OUTPUT_DIR / "curous_tokens.txt", "w", encoding="utf-8") as f:
        for tokens in docs_tokens:
            f.write(" ".join(tokens) + "\n")

    with open(OUTPUT_DIR / "curous_tokens.txt", "w", encoding="utf-8") as f:
        json.dump(docs_tokens, f)

    print(f"Docs kept (Category==1): {len(docs_tokens)}")
    print(f"Saved: {OUTPUT_DIR / 'corpus_tokens.txt'}")
    print(f"Saved: {OUTPUT_DIR / 'corpus_tokens.json'}")

if __name__ == "__main__":
    main()
    