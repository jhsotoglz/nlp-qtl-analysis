# This is the preprocessing script file that completes all the data preprocessing requirements
# File path: scripts/preprocess.py

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
    # Loading the QTL_text.json dataset. 
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)     # This will return a List[dict] with Keys: PMID, Journal, Title, Abstract, and Category fields
    
def preprocess_text(text: str, stop_word_set: set[str]) -> list[str]:
    # Pre-processing requirements: Split sentences, tokenize, lowercase, remove stop words and unwanted characters.
    tokens: list[str] = []
    for sentence in sent_tokenize(text):
        for token in word_tokenize(sentence):
            word = token.lower()
            if word.isalpha() and word not in stop_word_set:
                tokens.append(word)
    return tokens

def ensure_nltk_data():
    # Ensure required NLTK resources (punkt tokenizer + stopwords list) are downloaded
    try: 
        nltk.data.find("tokenizers/punkt")
    except LookupError: 
        nltk.download("punkt", quiet=True)

    try: 
        nltk.data.find("corpora/stopwords")
    except LookupError: 
        nltk.download("stopwords", quiet=True)

def main():
    # Make sure the outputs/ folder exists or create it if not
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    
    ensure_nltk_data()

    # Load English stop words into a set for fast checks
    stop_word_set = set(stopwords.words("english"))

    # Load the full dataset from QTL_text.json
    data = load_qtl_json(QTL_JSON)

    # Keep only abstracts where Category == "1"
    abstracts = [d.get("Abstract", "") for d in data if str(d.get("Category", "")).strip() == "1"]

    # Preprocess each abstract: sentence split → tokenize → lowercase → remove stopwords/non-alpha
    tokenized_abstracts = [preprocess_text(abs_text, stop_word_set) for abs_text in abstracts]

    # Save as a plain text file: one document per line, tokens separated by spaces
    txt_path = OUTPUT_DIR / "corpus_tokens.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for toks in tokenized_abstracts:
            f.write(" ".join(toks) + "\n")

    # Save as JSON: a list of lists (each inner list = tokens for one abstract)
    json_path = OUTPUT_DIR / "corpus_tokens.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(tokenized_abstracts, f)

    # Print summary info for quick verification
    print(f"Docs kept (Category==1): {len(tokenized_abstracts)}")
    print(f"Saved: {txt_path}")
    print(f"Saved: {json_path}")

if __name__ == "__main__":
    main()