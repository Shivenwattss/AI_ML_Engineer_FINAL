from pathlib import Path
import re
import json
import shutil
import os
import sys
import subprocess
import webbrowser
import threading
import time
import uuid
import pandas as pd

try:
    import requests  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - handled gracefully at runtime
    requests = None

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.memory import retrieve, save_experiment, load
from app.llm_agent import plan
from app.ml.engine import build


app = FastAPI(title="AI ML Engineer FINAL")


# Store generated models separately by ID.
# Models are loaded from their saved folders when requested.

def get_generated_model(model_id: str):
    import joblib

    # Only allow simple IDs, not paths
    if not re.fullmatch(r"[a-f0-9]{8}", model_id):
        raise HTTPException(status_code=404, detail="Model not found")

    model_dir = Path("generated_models") / model_id
    model_file = model_dir / "model.joblib"
    metadata_file = model_dir / "metadata.json"

    if not model_file.exists() or not metadata_file.exists():
        raise HTTPException(status_code=404, detail="Model not found")

    bundle = joblib.load(model_file)
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))

    return bundle, metadata


@app.get("/models/{model_id}")
def generated_model_page(model_id: str):
    if not re.fullmatch(r"[a-f0-9]{8}", model_id):
        raise HTTPException(status_code=404, detail="Model not found")

    page = Path("generated_models") / model_id / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Model page not found")

    return FileResponse(page)


@app.get("/api/models/{model_id}")
def generated_model_info(model_id: str):
    bundle, metadata = get_generated_model(model_id)
    return {
        "model_id": model_id,
        "model": metadata["model"],
        "task": metadata["task"],
        "target": metadata["target"],
        "features": metadata["features"],
        "feature_types": metadata["feature_types"],
        "docs_url": "/docs"
    }


@app.post("/api/models/{model_id}/predict")
async def generated_model_predict(model_id: str, payload: dict):
    import joblib
    import pandas as pd

    bundle, metadata = get_generated_model(model_id)
    pipeline = bundle["pipeline"]
    features = metadata["features"]

    missing = [feature for feature in features if feature not in payload]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required features: {missing}"
        )

    try:
