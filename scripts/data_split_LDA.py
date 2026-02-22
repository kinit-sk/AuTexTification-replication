import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from pathlib import Path
from nltk.corpus import stopwords

LANG = "en"
SUBTASK = "subtask_2"
DATA_DIR = Path("data/data/train") / SUBTASK / LANG
OUT_PATH = DATA_DIR / "train_5folds.tsv"

N_TOPICS = 10
SEED = 10

train_tsv = DATA_DIR / "train.tsv"
train_df = pd.read_csv(train_tsv, sep="\t")

if "text" not in train_df.columns:
    raise ValueError("Missing column 'text' in train.tsv")
if "id" not in train_df.columns:
    train_df["id"] = [f"train_{i}" for i in range(len(train_df))]

texts = train_df["text"].astype(str).tolist()

#stopwords based on language
if LANG == "en":
    stop_lang = "english"
elif LANG == "es":
    stop_lang = stopwords.words("spanish")
else:
    stop_lang = None

vectorizer = CountVectorizer(stop_words=stop_lang)
X = vectorizer.fit_transform(texts)

lda = LatentDirichletAllocation(
    n_components=N_TOPICS,
    random_state=SEED,
    learning_method="batch",
    max_iter=10,
)
lda_topics = lda.fit_transform(X)

topic_assignments = np.argmax(lda_topics, axis=1)
train_df["topic"] = topic_assignments

topic_sizes = train_df["topic"].value_counts().sort_values(ascending=False)
topic_order = topic_sizes.index.tolist()

pairs = []
for i in range(len(topic_order) // 2):
    pairs.append((topic_order[i], topic_order[-(i + 1)]))

topic_to_fold = {}
for fold_id, (t1, t2) in enumerate(pairs):
    topic_to_fold[t1] = fold_id
    topic_to_fold[t2] = fold_id

if len(topic_order) % 2 != 0:
    topic_to_fold[topic_order[len(topic_order) // 2]] = len(pairs)

train_df["fold"] = train_df["topic"].map(topic_to_fold)

fold_df = train_df[["id", "fold"]]
fold_df.to_csv(OUT_PATH, sep="\t", header=False, index=False)

print(f"Saved topic-based folds to: {OUT_PATH}")
print(fold_df['fold'].value_counts().sort_index())
