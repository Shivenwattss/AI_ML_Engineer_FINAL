from pathlib import Path
import shutil

def generate(model_path, out, task, target, features):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy2(model_path, out / "model.joblib")

    args = ",\n    ".join(f'"{x}": (object, ...)' for x in features)
    app_code = """from fastapi import FastAPI
from pydantic import create_model
import pandas as pd
import joblib

app = FastAPI(title="Generated ML Model API")
bundle = joblib.load("model.joblib")
pipeline = bundle["pipeline"]

Input = create_model("PredictionInput", **{
ARGS
})

@app.get("/")
def home():
    return {"task": TASK, "target": TARGET}

@app.post("/predict")
def predict(data: Input):
    row = pd.DataFrame([data.model_dump()])
    p = pipeline.predict(row)[0]
    return {"prediction": p.item() if hasattr(p, "item") else p}
""".replace("ARGS", args).replace("TASK", repr(task)).replace("TARGET", repr(target))
    (out / "app.py").write_text(app_code)
    (out / "requirements.txt").write_text(
        "fastapi\nuvicorn[standard]\npandas\nscikit-learn\nxgboost\njoblib\npydantic\n"
    )
    (out / "README.md").write_text(
        "Run: pip install -r requirements.txt\n"
        "Then: uvicorn app:app --reload\n"
    )
    return out
