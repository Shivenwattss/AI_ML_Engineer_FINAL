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
        row = pd.DataFrame([
            {feature: payload[feature] for feature in features}
        ])
        prediction = pipeline.predict(row)[0]

        if hasattr(prediction, "item"):
            prediction = prediction.item()

        if prediction is not None and not isinstance(
            prediction, (str, int, float, bool)
        ):
            prediction = str(prediction)

        result = {
            "model_id": model_id,
            "model": metadata["model"],
            "task": metadata["task"],
            "target": metadata["target"],
            "prediction": prediction
        }

        if metadata["task"] == "classification" and hasattr(
            pipeline, "predict_proba"
        ):
            probabilities = pipeline.predict_proba(row)[0]
            model = pipeline
            if hasattr(pipeline, "steps") and pipeline.steps:
                model = pipeline.steps[-1][1]

            classes = getattr(model, "classes_", None)
            if classes is not None:
                result["probabilities"] = {
                    str(label): float(probability)
                    for label, probability in zip(classes, probabilities)
                }

        return result

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

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

def gen_api(
    model_path,
    features,
    task,
    target,
    feature_types=None,
    selected_model="Selected ML Model",
    model_id=None
):
    if not model_id:
        model_id = uuid.uuid4().hex[:8]

    out = Path("generated_models") / model_id
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, out / "model.joblib")

    feature_types = feature_types or {}
    metadata = {
        "model_id": model_id,
        "model": selected_model,
        "task": task,
        "target": target,
        "features": list(features),
        "feature_types": feature_types
    }
    (out / "metadata.json").write_text(
        json.dumps(metadata, default=str),
        encoding="utf-8"
    )

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
async function initialise(){try{const response=await fetch('/api/models/MODEL_ID_PLACEHOLDER');if(!response.ok)throw new Error('Could not load model information.');const info=await response.json();document.getElementById('model-title').textContent=info.model||'Your prediction model';document.getElementById('task-pill').textContent='Task: '+readable(info.task);document.getElementById('target-pill').textContent='Target: '+info.target;document.getElementById('status-pill').textContent='API connected';fields.innerHTML='';fields.style.display='grid';fields.style.gridTemplateColumns='repeat(auto-fit,minmax(220px,1fr))';fields.style.gap='16px';for(const feature of info.features){const label=document.createElement('label');label.textContent=readable(feature);const input=document.createElement('input');input.name=feature;input.required=true;input.autocomplete='off';const dtype=String((info.feature_types||{})[feature]||'').toLowerCase();const numeric=dtype.includes('int')||dtype.includes('float')||dtype.includes('double');input.type=numeric?'number':'text';if(numeric)input.step=dtype.includes('int')?'1':'any';input.placeholder='Enter '+readable(feature).toLowerCase();label.appendChild(input);fields.appendChild(label)}button.disabled=false}catch(error){fields.textContent='Could not load the model form: '+error.message;document.getElementById('status-pill').textContent='API unavailable'}}
form.addEventListener('submit',async event=>{event.preventDefault();button.disabled=true;button.textContent='Predicting…';resultBox.style.display='none';try{const payload={};for(const [key,value] of new FormData(form).entries()){const input=form.elements.namedItem(key);payload[key]=input&&input.type==='number'?Number(value):value}const response=await fetch('/api/models/MODEL_ID_PLACEHOLDER/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const data=await response.json();if(!response.ok)throw new Error(data.detail||'Prediction failed.');resultBox.style.display='block';document.getElementById('prediction').className='prediction';document.getElementById('prediction').textContent=data.result?data.result+' — '+String(data.prediction):String(data.prediction);details.textContent=JSON.stringify(data,null,2)}catch(error){showError(error.message)}finally{button.disabled=false;button.textContent='Predict'}});initialise();
</script></body></html>'''
    html = html.replace("MODEL_ID_PLACEHOLDER", model_id)
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

        model_id = uuid.uuid4().hex[:8]

        generated = gen_api(
            model_path,
            features,
            task,
            target,
            feature_types=feature_types,
            selected_model=selected,
            model_id=model_id
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

        # Generated models are served by this main FastAPI process on Render.
        # No separate local-only server is started here.

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

            "model_id": model_id,
            "generated_website_url": f"/models/{model_id}",
            "generated_api_url": f"/api/models/{model_id}/predict",
            "generated_api_docs": "/docs"
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
async def chat(message: str = Form(...)):
    import os
    import joblib

    info = {}

    # Get information about the current trained model
    if Path("models/best_model.joblib").exists():
        b = joblib.load("models/best_model.joblib")
        info = {
            "task": b.get("task"),
            "target": b.get("target"),
            "features": b.get("features", [])
        }

    try:
        api_key = os.getenv("OLLAMA_API_KEY")

        if not api_key:
            return {
                "answer": "Ollama Cloud is not configured. Check OLLAMA_API_KEY in Render Environment.",
                "llm_connected": False
            }

        # Prepare the chat prompt
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

        # Send the request to Ollama Cloud
        r = requests.post(
            "https://ollama.com/api/chat",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": os.getenv("OLLAMA_CLOUD_MODEL", "gemma4:31b"),
                "messages": [
                    {"role": "user", "content": p}
                ],
                "stream": False
            },
            timeout=(10, 120)
        )

        r.raise_for_status()
        data = r.json()
        answer = data["message"]["content"]

        return {
            "answer": answer,
            "llm_connected": True,
            "llm": "Ollama Cloud"
        }

    except requests.RequestException as e:
        return {
            "answer": (
                "Could not connect to Ollama Cloud. "
                f"Error: {str(e)[:300]}"
            ),
            "llm_connected": False
        }

    except (ValueError, KeyError, TypeError) as e:
        return {
            "answer": f"Unexpected Ollama response: {str(e)[:300]}",
            "llm_connected": False
        }

    except Exception as e:
        return {
            "answer": f"Chat error: {type(e).__name__}: {str(e)[:300]}",
            "llm_connected": False
        }

