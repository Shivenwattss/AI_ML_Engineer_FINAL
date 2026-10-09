
import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:1.5b"

ALLOWED = {
    "classification": [
        "logistic_regression", "random_forest", "extra_trees",
        "hist_gradient_boosting", "svm", "knn", "xgboost"
    ],
    "regression": [
        "linear_regression", "ridge", "random_forest",
        "extra_trees", "hist_gradient_boosting", "svm",
        "knn", "xgboost"
    ],
    "anomaly_detection": ["isolation_forest"],
    "clustering": ["kmeans", "dbscan"]
}


def plan(problem, analysis, task, memories):
    fallback = ALLOWED.get(task, [])

    if not fallback:
        return {
            "candidate_models": [],
            "reason": f"Unsupported task: {task}"
        }

    # Keep the prompt compact.
    dataset = {
        "rows": analysis.get("rows"),
        "columns": analysis.get("columns"),
        "numeric": analysis.get("numeric_columns", [])[:15],
        "categorical": analysis.get("categorical_columns", [])[:10],
        "missing_values": analysis.get("missing_values")
    }

    prompt = (
        "You are an AutoML model selector. "
        "Choose at most 3 models from the allowed list. "
        "Return only JSON with keys candidate_models and reason. "
        "Do not invent model names.\n"
        f"Task: {task}\n"
        f"Problem: {str(problem)[:300]}\n"
        f"Dataset: {json.dumps(dataset, separators=(',', ':'))}\n"
        f"Allowed: {json.dumps(fallback)}\n"
        'Format: {"candidate_models":["model_name"],"reason":"short"}'
    )

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "keep_alive": "10m",
                "options": {
                    "temperature": 0,
                    "num_predict": 100
                }
            },
            timeout=(5, 120)
        )

        response.raise_for_status()
        result = response.json().get("response", "").strip()
        data = json.loads(result)

        candidates = [
            model for model in data.get("candidate_models", [])
            if model in fallback
        ]

        # Remove duplicates and cap the number of suggestions.
        candidates = list(dict.fromkeys(candidates))[:3]

        if not candidates:
            candidates = fallback[:3]

        return {
            "candidate_models": candidates,
            "reason": str(
                data.get("reason", "Candidates suggested by Ollama.")
            )[:300]
        }

    except (requests.RequestException, ValueError, KeyError, TypeError) as e:
        return {
            "candidate_models": fallback,
            "reason": (
                "Ollama recommendation unavailable; local evaluation "
                f"will choose the best model. ({type(e).__name__})"
            )
        }
