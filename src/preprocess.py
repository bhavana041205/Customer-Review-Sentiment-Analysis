"""
preprocess.py
-------------
Text cleaning utilities for the review dataset:
- lowercasing
- removing URLs, punctuation, digits
- removing stopwords
- light stemming (Porter stemmer, no external downloads needed)
"""

import re
from nltk.stem import PorterStemmer

STEMMER = PorterStemmer()

# Small built-in stopword list so we don't depend on nltk.download() at runtime.
STOPWORDS = set("""
a an the this that these those is are was were be been being
i you he she it we they me him her us them my your his its our their
and or but if while as of at by for with about against between into
through during before after above below to from up down in out on off
over under again further then once here there all any both each few
more most other some such no nor not only own same so than too very
s t can will just don should now
""".split())


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"[^a-z\s]", " ", text)                   # punctuation/digits
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    tokens = [STEMMER.stem(w) for w in tokens]
    return " ".join(tokens)


if __name__ == "__main__":
    sample = "This blender is AMAZING!! It exceeded my expectations, definately buying again :)"
    print(clean_text(sample))
