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


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

for x in ["data", "models", "memory", "generated_models"]:
    Path(x).mkdir(exist_ok=True)


# ============================================================
# STATIC FRONTEND
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


@app.get("/")
def home():
    return FileResponse("static/index.html")


# ============================================================
# GENERATED FASTAPI PROCESS
# ============================================================

generated_process = None


def start_generated_api():
    """Start the generated ML API on port 8001 and open its prediction website."""
    global generated_process

    if generated_process is not None and generated_process.poll() is None:
        try:
            generated_process.terminate()
            generated_process.wait(timeout=3)
        except Exception:
            try:
                generated_process.kill()
            except Exception:
                pass

    generated_dir = Path("generated_models/latest")
    if not generated_dir.exists():
        return

    generated_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8001"],
        cwd=str(generated_dir)
    )

    def open_browser():
        time.sleep(2)
        webbrowser.open("http://127.0.0.1:8001/")

    threading.Thread(target=open_browser, daemon=True).start()


# ============================================================
# DATA LOADING
# ============================================================

def load_data(path):
    e = Path(path).suffix.lower()

    if e == ".csv":
        return pd.read_csv(path)

    if e in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    if e == ".json":
        return pd.read_json(path)

    if e == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(
        "Supported files: CSV, Excel, JSON, Parquet."
    )


# ============================================================
# DATASET ANALYSIS
# ============================================================

def analyze(df):
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "columns_list": list(df.columns),

        "numeric_columns": list(
            df.select_dtypes(include="number").columns
        ),

        "categorical_columns": list(
            df.select_dtypes(exclude="number").columns
        ),

        "missing_values": int(
            df.isna().sum().sum()
        ),

        "duplicate_rows": int(
            df.duplicated().sum()
        ),

        "dtypes": {
            c: str(df[c].dtype)
            for c in df.columns
        }
    }


# ============================================================
# TARGET COLUMN DETECTION
# ============================================================

def target_guess(df, p):

    p = p.lower()

    m = re.search(
        r"(?:target|label|output)\s*(?:column)?\s*(?:is|=|:)\s*['\"]?([a-zA-Z_][\w .-]*)",
        p
    )

    if m:
        for c in df.columns:

            if c.lower() == m.group(1).strip().rstrip(" .").lower():
                return c

    for c in df.columns:

        if c.lower() in [
            "target",
            "label",
            "class",
            "output",
            "failure",
            "failed",
            "churn",
            "y"
        ]:
            return c

    if any(
        w in p
        for w in [
            "predict",
            "classify",
            "forecast",
            "whether"
        ]
    ):
        return df.columns[-1]

    return None


# ============================================================
# TASK DETECTION
# ============================================================

def task_guess(p, df, target):

    p = p.lower()

    if any(
        w in p
        for w in [
            "cluster",
            "segment",
            "group similar"
        ]
    ):
        return "clustering"

    if any(
        w in p
        for w in [
            "anomaly",
            "outlier",
            "fraud",
            "unusual",
            "abnormal"
        ]
    ):
        return "anomaly_detection"

    if target:

        if any(
            w in p
            for w in [
                "classify",
                "classification",
                "whether",
                "category",
                "yes or no"
            ]
        ):
            return "classification"

        if any(
            w in p
            for w in [
                "price",
                "amount",
                "revenue",
                "sales",
                "value",
                "number",
                "numeric"
            ]
        ):
            return "regression"

        return (
            "classification"
            if df[target].dtype == "object"
            or df[target].nunique() <= 20
            else "regression"
        )

    return "anomaly_detection"


# ============================================================
# GENERATE FASTAPI FOR SELECTED MODEL
# ============================================================

def gen_api(model_path, features, task, target, feature_types=None, selected_model="Selected ML Model"):
    # Create a standalone prediction website and FastAPI for the selected model.
    out = Path("generated_models/latest")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy2(model_path, out / "model.joblib")

    feature_types = feature_types or {}
    generated_code = r'''from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import create_model
import pandas as pd
import joblib

app = FastAPI(title="Generated ML Model API", version="1.0.0")
bundle = joblib.load("model.joblib")
pipeline = bundle["pipeline"]
FEATURES = FEATURES_VALUE
FEATURE_TYPES = FEATURE_TYPES_VALUE
TASK = TASK_VALUE
TARGET = TARGET_VALUE
MODEL_NAME = MODEL_NAME_VALUE
Input = create_model("PredictionInput", **{feature: (Any, ...) for feature in FEATURES})

def safe_value(value):
    if hasattr(value, "item"):
        value = value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

@app.get("/")
def home():
    return FileResponse("index.html")

@app.get("/health")
def health():
    return {"status": "healthy", "task": TASK, "target": TARGET, "model": MODEL_NAME}

@app.get("/model-info")
def model_info():
    return {"task": TASK, "target": TARGET, "model": MODEL_NAME, "features": FEATURES, "feature_types": FEATURE_TYPES, "docs_url": "/docs"}

@app.post("/predict")
def predict(data: Input):
    try:
        payload = data.model_dump()
        row = pd.DataFrame([{feature: payload[feature] for feature in FEATURES}])
        prediction = safe_value(pipeline.predict(row)[0])
        result = {"prediction": prediction, "task": TASK, "target": TARGET, "model": MODEL_NAME}

        if TASK == "anomaly_detection":
            if prediction == -1:
                result.update({"result": "Anomaly detected", "is_anomaly": True})
            elif prediction == 1:
                result.update({"result": "Normal", "is_anomaly": False})

        if TASK == "classification" and hasattr(pipeline, "predict_proba"):
            try:
                probabilities = pipeline.predict_proba(row)[0]
                classes = getattr(pipeline, "classes_", None)
                if classes is None and hasattr(pipeline, "steps") and pipeline.steps:
                    classes = getattr(pipeline.steps[-1][1], "classes_", None)
                if classes is not None:
                    result["probabilities"] = {str(safe_value(label)): float(probability) for label, probability in zip(classes, probabilities)}
            except Exception:
                pass
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
'''
    replacements = {
        "FEATURES_VALUE": repr(list(features)),
        "FEATURE_TYPES_VALUE": repr(feature_types),
        "TASK_VALUE": repr(task),
        "TARGET_VALUE": repr(target),
        "MODEL_NAME_VALUE": repr(selected_model),
    }
    for placeholder, value in replacements.items():
        generated_code = generated_code.replace(placeholder, value)
    (out / "app.py").write_text(generated_code, encoding="utf-8")

    html = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ML Model Prediction</title><style>
:root{color-scheme:light;--ink:#172033;--muted:#64748b;--line:#e2e8f0;--accent:#4f46e5}*{box-sizing:border-box}body{margin:0;background:#f5f7fb;color:var(--ink);font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}main{width:min(920px,calc(100% - 32px));margin:42px auto}header,.panel{background:#fff;border:1px solid var(--line);border-radius:18px;padding:24px;box-shadow:0 8px 28px #1720330a}header{margin-bottom:18px}.eyebrow{color:var(--accent);font-weight:700;text-transform:uppercase;letter-spacing:.08em;font-size:12px}h1{margin:8px 0;font-size:clamp(26px,4vw,36px)}.muted{color:var(--muted);margin:0}.meta{display:flex;gap:8px;flex-wrap:wrap;margin-top:18px}.pill{border:1px solid var(--line);border-radius:999px;padding:5px 10px;font-size:13px;background:#fafbff}h2{font-size:20px;margin:0 0 16px}form{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}label{display:block;font-size:14px;font-weight:650}input{width:100%;margin-top:6px;padding:11px 12px;border:1px solid #cbd5e1;border-radius:10px;font:inherit}input:focus{outline:2px solid #c7d2fe;border-color:var(--accent)}.actions{grid-column:1/-1;display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-top:4px}button{border:0;border-radius:10px;padding:12px 18px;background:var(--accent);color:#fff;font:inherit;font-weight:700;cursor:pointer}button:disabled{opacity:.6;cursor:wait}.result{margin-top:18px;display:none}.prediction{font-size:26px;font-weight:750;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f8fafc;padding:14px;border-radius:10px}a{color:var(--accent);font-weight:650}.error{color:#b91c1c}footer{text-align:center;color:var(--muted);font-size:13px;padding:18px}
</style></head><body><main>
<header><div class="eyebrow">Generated machine learning app</div><h1 id="model-title">Your prediction model</h1><p class="muted">Enter the feature values below, then select Predict.</p><div class="meta"><span class="pill" id="task-pill">Task: loading…</span><span class="pill" id="target-pill">Target: loading…</span><span class="pill" id="status-pill">Checking API…</span></div></header>
<section class="panel"><h2>Make a prediction</h2><form id="prediction-form"><div id="fields" style="grid-column:1/-1">Loading model features…</div><div class="actions"><button id="predict-button" type="submit" disabled>Predict</button><a href="/docs" target="_blank" rel="noopener">Open FastAPI Swagger docs</a></div></form>
<section class="result panel" id="result" aria-live="polite"><h2>Prediction result</h2><div class="prediction" id="prediction"></div><pre id="details"></pre></section></section><footer>Predictions are generated by the selected trained model. Check input values before using results for real decisions.</footer></main>
<script>
const form=document.getElementById('prediction-form'),fields=document.getElementById('fields'),button=document.getElementById('predict-button'),resultBox=document.getElementById('result'),details=document.getElementById('details');
function readable(v){return String(v).replaceAll('_',' ').replace(/\b\w/g,c=>c.toUpperCase())}
function showError(message){resultBox.style.display='block';document.getElementById('prediction').textContent='Could not predict';document.getElementById('prediction').className='prediction error';details.textContent=message}
async function initialise(){try{const response=await fetch('/model-info');if(!response.ok)throw new Error('Could not load model information.');const info=await response.json();document.getElementById('model-title').textContent=info.model||'Your prediction model';document.getElementById('task-pill').textContent='Task: '+readable(info.task);document.getElementById('target-pill').textContent='Target: '+info.target;document.getElementById('status-pill').textContent='API connected';fields.innerHTML='';fields.style.display='grid';fields.style.gridTemplateColumns='repeat(auto-fit,minmax(220px,1fr))';fields.style.gap='16px';for(const feature of info.features){const label=document.createElement('label');label.textContent=readable(feature);const input=document.createElement('input');input.name=feature;input.required=true;input.autocomplete='off';const dtype=String((info.feature_types||{})[feature]||'').toLowerCase();const numeric=dtype.includes('int')||dtype.includes('float')||dtype.includes('double');input.type=numeric?'number':'text';if(numeric)input.step=dtype.includes('int')?'1':'any';input.placeholder='Enter '+readable(feature).toLowerCase();label.appendChild(input);fields.appendChild(label)}button.disabled=false}catch(error){fields.textContent='Could not load the model form: '+error.message;document.getElementById('status-pill').textContent='API unavailable'}}
form.addEventListener('submit',async event=>{event.preventDefault();button.disabled=true;button.textContent='Predicting…';resultBox.style.display='none';try{const payload={};for(const [key,value] of new FormData(form).entries()){const input=form.elements.namedItem(key);payload[key]=input&&input.type==='number'?Number(value):value}const response=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const data=await response.json();if(!response.ok)throw new Error(data.detail||'Prediction failed.');resultBox.style.display='block';document.getElementById('prediction').className='prediction';document.getElementById('prediction').textContent=data.result?data.result+' — '+String(data.prediction):String(data.prediction);details.textContent=JSON.stringify(data,null,2)}catch(error){showError(error.message)}finally{button.disabled=false;button.textContent='Predict'}});initialise();
</script></body></html>'''
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "requirements.txt").write_text(
        "fastapi\nuvicorn[standard]\npandas\nscikit-learn\nxgboost\njoblib\npydantic\n",
        encoding="utf-8"
    )
    return out


# ============================================================
# BUILD MODEL
# ============================================================

@app.post("/build")
async def build_model(
    problem: str = Form(...),
    file: UploadFile = File(...),
    target: str = Form("")
):

    # Check file type
    if Path(file.filename).suffix.lower() not in [
        ".csv",
        ".xlsx",
        ".xls",
        ".json",
        ".parquet"
    ]:
        raise HTTPException(
            400,
            "Supported: CSV, Excel, JSON, Parquet."
        )

    # Save uploaded dataset
    path = Path("data") / file.filename

    path.write_bytes(
        await file.read()
    )

    try:

        # ====================================================
        # LOAD DATA
        # ====================================================

        df = load_data(path)

        if df.empty or len(df.columns) < 2:

            raise ValueError(
                "Dataset must have data and at least two columns."
            )

        # ====================================================
        # DETECT TARGET + TASK
        # ====================================================

        target = (
            target.strip()
            or target_guess(df, problem)
        )

        task = task_guess(
            problem,
            df,
            target
        )

        # ====================================================
        # ANALYZE DATA
        # ====================================================

        analysis = analyze(df)

        # ====================================================
        # RETRIEVE PREVIOUS EXPERIMENTS
        # ====================================================

        mem = retrieve(
            problem
            + " "
            + task
            + " "
            + " ".join(map(str, df.columns)),
            3
        )

        # ====================================================
        # ASK OLLAMA / QWEN FOR MODEL STRATEGY
        # ====================================================

        strategy = plan(
            problem,
            analysis,
            task,
            mem
        )

        # ====================================================
        # TRAIN + EVALUATE MODELS
        # ====================================================

        results, selected, model_path, features = build(
            df,
            target,
            task,
            strategy["candidate_models"]
        )

        # ====================================================
        # GENERATE FASTAPI
        # ====================================================

        feature_types = {
            feature: str(df[feature].dtype)
            for feature in features
            if feature in df.columns
        }

        generated = gen_api(
            model_path,
            features,
            task,
            target,
            feature_types=feature_types,
            selected_model=selected
        )

        # ====================================================
        # SAVE EXPERIMENT TO RAG MEMORY
        # ====================================================

        save_experiment(
            {
                "problem": problem,
                "task": task,
                "target": target,
                "columns": features,
                "selected_model": selected,
                "results": results,
                "reason": strategy["reason"]
            }
        )

        # ====================================================
        # AUTOMATICALLY START GENERATED FASTAPI
        # ====================================================

        start_generated_api()

        # ====================================================
        # RETURN RESULT
        # ====================================================

        return {
            "task": task,
            "target": target,

            "dataset_analysis": analysis,

            "llm_strategy": strategy,

            "retrieved_memories": mem,

            "results": results,

            "selected_model": selected,

            "model_path": str(
                model_path
            ),

            "generated_api_path": str(
                generated
            ),

            "generated_website_url":
                "http://127.0.0.1:8001/",

            "generated_api_url":
                "http://127.0.0.1:8001",

            "generated_api_docs":
                "http://127.0.0.1:8001/docs"
        }

    except Exception as e:

        raise HTTPException(
            400,
            str(e)
        )


# ============================================================
# RAG MEMORY
# ============================================================

@app.get("/memory")
def memory():

    return {
        "experiments": load()
    }


# ============================================================
# AI CHAT USING OLLAMA
# ============================================================

@app.post("/chat")
async def chat(
    message: str = Form(...)
):

    import joblib

    info = {}

    # ========================================================
    # CURRENT TRAINED MODEL
    # ========================================================

    if Path(
        "models/best_model.joblib"
    ).exists():

        b = joblib.load(
            "models/best_model.joblib"
        )

        info = {
            "task": b.get("task"),
            "target": b.get("target"),
            "features": b.get(
                "features",
                []
            )
        }

    try:

        # ====================================================
        # PROMPT FOR QWEN
        # ====================================================

        p = f"""
You are the AI ML Engineer assistant.

Current trained model:
{json.dumps(info, default=str)}

Previous experiments:
{json.dumps(load()[-5:], default=str)}

User question:
{message}

Answer clearly and only use information available in the context.
"""

        # ====================================================
        # SEND TO OLLAMA
        # ====================================================

        r = requests.post(
            "http://localhost:11434/api/generate",

            json={
                "model": "qwen2.5:1.5b",
                "prompt": p,
                "stream": False
            },

            timeout=120
        )

        r.raise_for_status()

        answer = r.json()["response"]

        return {
            "answer": answer,
            "llm_connected": True,
            "llm": "Ollama / Qwen 2.5:1.5B"
        }

    except Exception as e:

        return {
            "answer": str(e),
            "llm_connected": False
        }