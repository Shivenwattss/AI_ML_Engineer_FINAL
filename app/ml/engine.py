
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, r2_score, silhouette_score
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, IsolationForest
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans, DBSCAN
from xgboost import XGBClassifier, XGBRegressor
Path("models").mkdir(exist_ok=True)

CLS=["logistic_regression","random_forest","extra_trees","hist_gradient_boosting","svm","knn","xgboost"]
REG=["linear_regression","ridge","random_forest","extra_trees","hist_gradient_boosting","svm","knn","xgboost"]

def add_dates(df):
    out=df.copy()
    for c in list(out.columns):
        if out[c].dtype=="object":
            d=pd.to_datetime(out[c],errors="coerce")
            if d.notna().mean()>=.8:
                for n,v in [("_year",d.dt.year),("_month",d.dt.month),("_day",d.dt.day),("_dow",d.dt.dayofweek),("_hour",d.dt.hour)]:
                    out[c+n]=v
                out.drop(columns=[c],inplace=True)
    return out

def prep(X):
    nums=X.select_dtypes(include=np.number).columns.tolist()
    cats=[c for c in X.columns if c not in nums]
    t=[]
    if nums:t.append(("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]),nums))
    if cats:t.append(("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cats))
    if not t: raise ValueError("No usable feature columns.")
    return ColumnTransformer(t)

def mdl(n,t):
    if t=="classification":
        return {"logistic_regression":LogisticRegression(max_iter=2000),"random_forest":RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1),"extra_trees":ExtraTreesClassifier(n_estimators=200,random_state=42,n_jobs=-1),"hist_gradient_boosting":HistGradientBoostingClassifier(random_state=42),"svm":SVC(probability=True),"knn":KNeighborsClassifier(),"xgboost":XGBClassifier(n_estimators=150,max_depth=6,learning_rate=.08,random_state=42,eval_metric="logloss",n_jobs=-1)}[n]
    return {"linear_regression":LinearRegression(),"ridge":Ridge(),"random_forest":RandomForestRegressor(n_estimators=200,random_state=42,n_jobs=-1),"extra_trees":ExtraTreesRegressor(n_estimators=200,random_state=42,n_jobs=-1),"hist_gradient_boosting":HistGradientBoostingRegressor(random_state=42),"svm":SVR(),"knn":KNeighborsRegressor(),"xgboost":XGBRegressor(n_estimators=150,max_depth=6,learning_rate=.08,random_state=42,objective="reg:squarederror",n_jobs=-1)}[n]

def build(df,target,task,candidates=None):
    df=add_dates(df)
    if task=="clustering":
        X=df.select_dtypes(include=np.number).copy()
        if X.shape[1]<2: raise ValueError("Clustering needs at least two numeric columns.")
        X=X.replace([np.inf,-np.inf],np.nan).fillna(X.median())
        Xs=StandardScaler().fit_transform(X); results=[]; best=None
        for n in candidates or ["kmeans","dbscan"]:
            try:
                m=KMeans(n_clusters=3,random_state=42,n_init=10) if n=="kmeans" else DBSCAN()
                lab=m.fit_predict(Xs)
                s=float(silhouette_score(Xs,lab)) if len(set(lab))>1 and set(lab)!={-1} else None
                results.append({"model":n,"metric":"silhouette_score","score":round(s,4) if s is not None else None})
                if s is not None and (best is None or s>best[0]):best=(s,n,m)
            except Exception as e:results.append({"model":n,"metric":"failed","score":None,"error":str(e)})
        if best is None: raise RuntimeError("No clustering model produced a valid result.")
        path=Path("models/best_model.joblib"); joblib.dump({"pipeline":best[2],"features":X.columns.tolist(),"task":task,"target":None},path)
        return results,best[1],path,X.columns.tolist()
    if task=="anomaly_detection":
        X=df.select_dtypes(include=np.number).copy()
        if X.shape[1]==0: raise ValueError("Anomaly detection needs numeric columns.")
        X=X.replace([np.inf,-np.inf],np.nan).fillna(X.median())
        m=IsolationForest(contamination="auto",random_state=42);m.fit(X)
        path=Path("models/best_model.joblib");joblib.dump({"pipeline":m,"features":X.columns.tolist(),"task":task,"target":None},path)
        return [{"model":"isolation_forest","metric":"unsupervised_detector","score":None}],"isolation_forest",path,X.columns.tolist()
    if not target or target not in df.columns: raise ValueError("A target column is required for prediction.")
    X=df.drop(columns=[target]);y=df[target];mask=y.notna();X=X.loc[mask];y=y.loc[mask]
    strat=y if task=="classification" and y.nunique()>1 else None
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=strat)
    results=[];best=None;allowed=CLS if task=="classification" else REG
    for n in candidates or allowed:
        if n not in allowed:continue
        try:
            p=Pipeline([("preprocess",prep(Xtr)),("model",mdl(n,task))]);p.fit(Xtr,ytr);pred=p.predict(Xte)
            score=f1_score(yte,pred,average="weighted") if task=="classification" else r2_score(yte,pred)
            metric="weighted_f1" if task=="classification" else "r2"
            results.append({"model":n,"metric":metric,"score":round(float(score),4)})
            if best is None or score>best[0]:best=(score,n,p)
        except Exception as e:results.append({"model":n,"metric":"failed","score":None,"error":str(e)})
    if best is None:raise RuntimeError("No candidate model could be trained.")
    path=Path("models/best_model.joblib");joblib.dump({"pipeline":best[2],"features":X.columns.tolist(),"task":task,"target":target},path)
    return results,best[1],path,X.columns.tolist()
