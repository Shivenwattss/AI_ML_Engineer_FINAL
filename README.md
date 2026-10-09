# AI ML Engineer FINAL

Standalone final **tabular AutoML** build.

Supports CSV, Excel, JSON and Parquet; numeric/categorical data; missing values; common datetime columns; classification, regression, clustering and anomaly detection.

Models:
Classification: Logistic Regression, Random Forest, Extra Trees, HistGradientBoosting, SVM, KNN, XGBoost.
Regression: Linear Regression, Ridge, Random Forest, Extra Trees, HistGradientBoosting, SVM, KNN, XGBoost.
Unsupervised: KMeans, DBSCAN, Isolation Forest.

The LLM proposes candidates, real validation metrics choose the winner. It saves experiments to RAG memory, provides chat, saves the trained model and generates a FastAPI service.

Run:
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
$env:OPENAI_API_KEY="YOUR_REAL_API_KEY"
$env:OPENAI_MODEL="gpt-6-luna"
uvicorn app.main:app --reload

Open http://127.0.0.1:8000

Important: no system can honestly guarantee literally every possible data modality. This version targets common tabular ML data. Images/audio/video need dedicated deep-learning pipelines.
