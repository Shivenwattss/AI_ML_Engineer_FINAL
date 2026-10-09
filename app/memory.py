from pathlib import Path
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MEMORY = Path("memory/experiments.json")
MEMORY.parent.mkdir(exist_ok=True)

def load():
    if not MEMORY.exists():
        return []
    try:
        return json.loads(MEMORY.read_text())
    except Exception:
        return []

def save_experiment(record):
    items = load()
    items.append(record)
    MEMORY.write_text(json.dumps(items, indent=2, default=str))

def retrieve(query, k=3):
    items = load()
    if not items:
        return []
    texts = [
        " ".join([
            str(x.get("problem","")), str(x.get("task","")),
            str(x.get("columns","")), str(x.get("selected_model","")),
            str(x.get("reason",""))
        ]) for x in items
    ]
    try:
        v = TfidfVectorizer(stop_words="english")
        mat = v.fit_transform(texts + [query])
        sims = cosine_similarity(mat[-1], mat[:-1]).ravel()
        order = sims.argsort()[::-1][:k]
        return [{**items[i], "similarity": round(float(sims[i]), 3)}
                for i in order if sims[i] > 0]
    except Exception:
        return items[-k:]
