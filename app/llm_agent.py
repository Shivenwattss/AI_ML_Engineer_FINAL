
import json
import os
import requests

# Ollama Cloud configuration
OLLAMA_URL = "https://ollama.com/api/chat"
OLLAMA_MODEL = os.getenv("OLLAMA_CLOUD_MODEL", "gemma4:31b")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

# Models allowed for each task
ALLOWED = {
    "classification": [
        "logistic_regression",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "svm",
        "knn",
        "xgboost"
    ],
    "regression": [
        "linear_regression",
        "ridge",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "svm",
        "knn",
        "xgboost"
    ],
    "anomaly_detection": [
        "isolation_forest"
    ],
    "clustering": [
        "kmeans",
        "dbscan"
    ]
}


def plan(problem, analysis, task, memories):
    """
    Ask Ollama Cloud to recommend suitable candidate models.
    If the cloud request fails, return allowed fallback models.
    """

    fallback = ALLOWED.get(task, [])

    if not fallback:
        return {
            "candidate_models": [],
            "reason": f"Unsupported task: {task}"
        }

    # Keep the dataset summary compact
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
        "Return only valid JSON with keys candidate_models and reason. "
        "Do not invent model names.\n"
        f"Task: {task}\n"
        f"Problem: {str(problem)[:300]}\n"
        f"Dataset: {json.dumps(dataset, separators=(',', ':'))}\n"
        f"Allowed models: {json.dumps(fallback)}\n"
        'Format: {"candidate_models":["model_name"],"reason":"short"}'
    )

    # Keep the app usable when no API key is configured
    if not OLLAMA_API_KEY:
        return {
            "candidate_models": fallback[:3],
            "reason": (
                "Ollama Cloud API key is missing. "
                "The local evaluation process will select the best model."
            )
        }

    try:
        response = requests.post(
            OLLAMA_URL,
            headers={
                "Authorization": f"Bearer {OLLAMA_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0,
                    "num_predict": 150
                }
            },
            timeout=(10, 120)
        )

        response.raise_for_status()

        result = response.json()["message"]["content"].strip()
        data = json.loads(result)

        # Accept only models from the approved list
        candidates = [
            model
            for model in data.get("candidate_models", [])
            if isinstance(model, str) and model in fallback
        ]

        # Remove duplicates and limit to 3 suggestions
        candidates = list(dict.fromkeys(candidates))[:3]

        if not candidates:
            candidates = fallback[:3]

        return {
            "candidate_models": candidates,
            "reason": str(
                data.get(
                    "reason",
                    "Candidate models suggested by Ollama Cloud."
                )
            )[:300]
        }

    except (
        requests.RequestException,
        ValueError,
        KeyError,
        TypeError
    ) as e:
        return {
            "candidate_models": fallback[:3],
            "reason": (
                "Ollama Cloud recommendation unavailable. "
                "The local evaluation process will select the best model. "
                f"({type(e).__name__})"
            )
        }
