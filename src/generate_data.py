"""
generate_data.py
-----------------
Generates a synthetic but realistic customer-review dataset for sentiment
analysis. Real scraped datasets (Amazon/Yelp) can't be pulled in this
environment, so reviews are built from templates + a large vocabulary bank,
with injected noise (typos, negation, mixed-sentiment, varied length) so the
classification task is not trivially easy.

Output: data/reviews.csv  (columns: review_text, rating, sentiment)
"""

import random
import re
import csv

random.seed(42)

CATEGORIES = [
    "headphones", "laptop", "blender", "running shoes", "backpack",
    "coffee maker", "smartwatch", "office chair", "bluetooth speaker",
    "vacuum cleaner", "air fryer", "phone case", "desk lamp", "yoga mat",
    "wireless mouse",
]

POSITIVE_ADJ = [
    "excellent", "amazing", "fantastic", "great", "wonderful", "superb",
    "impressive", "outstanding", "solid", "reliable", "durable", "sturdy",
    "comfortable", "sleek", "smooth", "fast", "responsive", "affordable",
    "worth every penny", "well made", "top notch",
]

NEGATIVE_ADJ = [
    "terrible", "awful", "disappointing", "poor", "cheap", "flimsy",
    "unreliable", "slow", "clunky", "overpriced", "uncomfortable",
    "defective", "broken", "faulty", "subpar", "frustrating", "useless",
    "shoddy", "not worth it", "badly made",
]

NEUTRAL_ADJ = [
    "okay", "decent", "average", "fine", "acceptable", "so-so",
    "reasonable", "standard", "nothing special", "middle of the road",
]

POSITIVE_CLAUSES = [
    "exceeded my expectations", "works perfectly out of the box",
    "the build quality is great", "customer service was very helpful",
    "shipping was fast and packaging was solid",
    "I would definitely buy this again", "battery life is impressive",
    "it feels premium for the price", "setup took less than five minutes",
    "my whole family loves it now",
]

NEGATIVE_CLAUSES = [
    "stopped working within a week", "the instructions were confusing",
    "customer support never responded", "it arrived damaged",
    "the material feels cheap", "I regret this purchase",
    "battery drains way too fast", "it's louder than advertised",
    "the size runs way smaller than expected",
    "I had to return it after two days",
]

NEUTRAL_CLAUSES = [
    "it does the job but nothing more", "packaging was fine",
    "delivery took about a week", "some features are useful, others aren't",
    "it's fine for the price but I wouldn't rave about it",
    "works as described, no surprises",
]

NEGATIONS = [
    ("not bad", "positive"), ("not great", "negative"),
    ("could be better", "negative"), ("not as good as I hoped", "negative"),
    ("didn't disappoint", "positive"), ("wouldn't recommend", "negative"),
    ("can't complain", "positive"),
]

TYPOS = {"the": "teh", "very": "vary", "quality": "qaulity",
         "recommend": "reccomend", "definitely": "definately",
         "product": "prodcut", "excellent": "excelent"}


def maybe_typo(text, p=0.12):
    words = text.split()
    for i, w in enumerate(words):
        lw = w.lower().strip(".,!")
        if lw in TYPOS and random.random() < p:
            words[i] = w.replace(lw, TYPOS[lw])
    return " ".join(words)


def build_review(sentiment):
    """Builds a review for the target sentiment, but deliberately blends in
    opposite-sentiment clauses ~25% of the time (mixed/hedged reviews) and
    occasionally omits the strong marker adjective, so the classes overlap
    in vocabulary the way real review text does."""
    cat = random.choice(CATEGORIES)
    parts = []
    all_adj = {"positive": POSITIVE_ADJ, "negative": NEGATIVE_ADJ, "neutral": NEUTRAL_ADJ}
    all_clauses = {"positive": POSITIVE_CLAUSES, "negative": NEGATIVE_CLAUSES, "neutral": NEUTRAL_CLAUSES}

    # 20% of the time, open with a neutral/mild framing instead of the strong marker
    if random.random() < 0.2:
        parts.append(f"Bought this {cat} a few weeks ago.")
    else:
        parts.append(f"This {cat} is {random.choice(all_adj[sentiment])}.")

    # main supporting clause, usually matching sentiment but sometimes mixed
    if random.random() < 0.75:
        parts.append(random.choice(all_clauses[sentiment]).capitalize() + ".")

    # 25% chance: inject a clause from a *different* sentiment (mixed opinion,
    # e.g. "great sound but the battery died fast") -- makes the task harder
    if random.random() < 0.25:
        other = random.choice([s for s in ["positive", "negative", "neutral"] if s != sentiment])
        connector = random.choice(["However,", "That said,", "On the downside," if other == "negative" else "On the plus side,", "Also,"])
        parts.append(f"{connector} {random.choice(all_clauses[other])}.")

    if random.random() < 0.15:
        neg_clause, s = random.choice(NEGATIONS)
        parts.append(f"Overall it's {neg_clause}.")

    if sentiment == "positive":
        rating = random.choice([4, 5, 5, 4, 3])  # occasional label noise via rating
    elif sentiment == "negative":
        rating = random.choice([1, 2, 2, 1, 3])
    else:
        rating = random.choice([2, 3, 3, 3, 4])

    if random.random() < 0.3:
        parts.append(f"Rating: {rating}/5.")

    # sometimes keep it very short (real reviews are often terse)
    if random.random() < 0.15:
        parts = parts[:1]

    text = " ".join(parts)
    text = maybe_typo(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text, rating


def main(n_total=10500, out_path="data/reviews.csv"):
    # Slightly imbalanced classes, like real review data (more positives)
    n_pos = int(n_total * 0.45)
    n_neg = int(n_total * 0.35)
    n_neu = n_total - n_pos - n_neg

    rows = []
    for _ in range(n_pos):
        text, rating = build_review("positive")
        rows.append((text, rating, "positive"))
    for _ in range(n_neg):
        text, rating = build_review("negative")
        rows.append((text, rating, "negative"))
    for _ in range(n_neu):
        text, rating = build_review("neutral")
        rows.append((text, rating, "neutral"))

    random.shuffle(rows)

    # Inject ~3% label noise, like real human-tagged review data has
    noisy_rows = []
    all_labels = ["positive", "negative", "neutral"]
    for text, rating, label in rows:
        if random.random() < 0.03:
            label = random.choice([l for l in all_labels if l != label])
        noisy_rows.append((text, rating, label))
    rows = noisy_rows

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["review_text", "rating", "sentiment"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} reviews to {out_path}")
    print(f"  positive: {n_pos}, negative: {n_neg}, neutral: {n_neu}")


if __name__ == "__main__":
    main()
