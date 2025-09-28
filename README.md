# COM S 5790 – NLP Project 1

This is the first project of the class COM S 5790, where we apply Natural Language Processing techniques to a collection of data. 
The goal is to practice key NLP methods such as word cloud visualization, Word2Vec embeddings, and phrase extraction.

## It follows the assignment tasks:

- **Preprocessing:**
  - Keep only abstracts with `Category == "1"`.
  - Sentence split.
  - Tokenize.
  - Lowercase.
  - Remove stopwords and non-alphabetic tokens.

- **Task 1:** Word cloud visualization
  - Frequency-based word cloud (`freq_wordcloud.png`)
  - TF-IDF–based word cloud (`tfidf_wordcloud.png`)

- **Task 2:** Train a Word2Vec model
  - Parameters: `vector_size=100`, `window=5`, `min_count=10`, `sg=0 (CBOW)`
  - Compute top-10 TF-IDF terms
  - For each, output top-20 most similar words (`task2_word2vec.txt`)

- **Task 3:** Phrase extraction + repeat Tasks 1 & 2
  - Learn bigrams/trigrams with Gensim’s `Phrases`
  - Save phrased corpus (`corpus_tokens_phrased.txt` / `.json`)
  - Generate frequency & TF-IDF word clouds (`phrased_freq_wordcloud.png`, `phrased_tfidf_wordcloud.png`)
  - Train Word2Vec on phrased corpus and output neighbors (`task3_word2vec_phrased.txt`)

---

## Repository Structure

```
outputs/           # All generated results (txt, json, png)
scripts/
  preprocess.py    
  task1_wordcloud.py
  task2_word2vec.py
  phrases.py     
  task3_wordcloud_phrased.py
  task3_word2vec_phrased.py
```

---

## How to Run

### Preprocessing
```bash
python scripts/preprocess.py
```
Outputs:
- `outputs/corpus_tokens.txt` tokens separated by space for each abstract
- `outputs/corpus_tokens.json` a list of token lists

---

### Task 1 – Word Clouds
```bash
python scripts/task1_wordclouds.py
```
Outputs:
- `outputs/freq_wordcloud_task1.png`
- `outputs/tfidf_wordcloud_task1.png`

---

### Task 2 – Word2Vec and TF-IDF Neighbors
```bash
python scripts/task2_word2vec.py
```
Outputs:
- `outputs/task2_word2vec.txt`

---

### Task 3 – Phrases and Repeat Previous Tasks
Get biagrams and trigrams:
```bash
python scripts/phrases.py
```

Generate word clouds:
```bash
python scripts/task3_wordclouds.py
```

Train Word2Vec and neighbors:
```bash
python scripts/task3_word2vec.py
```

Outputs:
- `outputs/corpus_tokens_phrased.txt`
- `outputs/corpus_tokens_phrased.json`
- `outputs/phrased_freq_wordcloud.png`
- `outputs/phrased_tfidf_wordcloud.png`
- `outputs/task3_word2vec_phrased.txt`