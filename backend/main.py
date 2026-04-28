"""
Heart Disease Prediction API
==============================
FastAPI application exposing REST endpoints for:
  - POST /predict        → prediction + SHAP explanation
  - GET  /meta           → feature metadata & model metrics
  - GET  /feature-importance → global SHAP importance
  - GET  /health         → liveness check

Run locally:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import json
import os
import pickle
import time
import warnings
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Globals (loaded once on startup)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_DIR = os.getenv("MODEL_DIR", "model")

_model     = None
_scaler    = None
_explainer = None
_meta: dict[str, Any] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan: load artifacts at startup
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _scaler, _explainer, _meta
    print("🚀 Loading model artifacts...")

    with open(f"{MODEL_DIR}/model.pkl",  "rb") as f: _model  = pickle.load(f)
    with open(f"{MODEL_DIR}/scaler.pkl", "rb") as f: _scaler = pickle.load(f)
    with open(f"{MODEL_DIR}/meta.json",  "r")  as f: _meta   = json.load(f)

    # Rebuild SHAP explainer (pickle may not be available)
    explainer_path = f"{MODEL_DIR}/explainer.pkl"
    if os.path.exists(explainer_path):
        try:
            with open(explainer_path, "rb") as f:
                _explainer = pickle.load(f)
            print("✅ SHAP explainer loaded from disk")
        except Exception:
            _explainer = _rebuild_explainer()
    else:
        _explainer = _rebuild_explainer()

    print("✅ All artifacts ready")
    yield
    print("🛑 Shutting down")


def _rebuild_explainer():
    """Reconstruct SHAP explainer from model type."""
    print("⚙️  Rebuilding SHAP explainer…")
    try:
        exp = shap.TreeExplainer(_model)
        print("✅ TreeExplainer ready")
        return exp
    except Exception:
        # KernelExplainer needs background data; use a small random sample
        bg = np.zeros((50, len(_meta["feature_names"])))
        exp = shap.KernelExplainer(_model.predict_proba, bg)
        print("✅ KernelExplainer ready (fallback)")
        return exp


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Heart Disease Prediction API",
    description="ML-powered cardiovascular risk assessment using the UCI Heart Disease dataset.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class PatientInput(BaseModel):
    """All 13 UCI Cleveland features — see /meta for human-readable labels."""
    age:      float = Field(..., ge=1,   le=120,  description="Age in years")
    sex:      int   = Field(..., ge=0,   le=1,    description="0=Female, 1=Male")
    cp:       int   = Field(..., ge=0,   le=3,    description="Chest pain type (0–3)")
    trestbps: float = Field(..., ge=50,  le=250,  description="Resting blood pressure (mm Hg)")
    chol:     float = Field(..., ge=50,  le=700,  description="Serum cholesterol (mg/dl)")
    fbs:      int   = Field(..., ge=0,   le=1,    description="Fasting blood sugar >120 mg/dl")
    restecg:  int   = Field(..., ge=0,   le=2,    description="Resting ECG results")
    thalach:  float = Field(..., ge=50,  le=250,  description="Max heart rate achieved (bpm)")
    exang:    int   = Field(..., ge=0,   le=1,    description="Exercise induced angina")
    oldpeak:  float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise")
    slope:    int   = Field(..., ge=0,   le=2,    description="Slope of peak exercise ST segment")
    ca:       int   = Field(..., ge=0,   le=4,    description="Number of major vessels colored")
    thal:     int   = Field(..., ge=0,   le=3,    description="Thalassemia type")

    model_config = {"json_schema_extra": {
        "example": {
            "age": 54, "sex": 1, "cp": 0, "trestbps": 122,
            "chol": 286, "fbs": 0, "restecg": 0, "thalach": 116,
            "exang": 1, "oldpeak": 3.2, "slope": 1, "ca": 2, "thal": 2
        }
    }}


class SHAPValue(BaseModel):
    feature: str
    value:   float
    impact:  float   # SHAP contribution


class PredictionResponse(BaseModel):
    prediction:   int            # 0 = No Disease, 1 = Disease
    probability:  float          # P(disease)
    risk_level:   str            # "Low" | "Moderate" | "High"
    confidence:   float          # max(P(0), P(1))
    shap_values:  list[SHAPValue]
    elapsed_ms:   float


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal"
]

def _risk_level(prob: float) -> str:
    if prob < 0.35:  return "Low"
    if prob < 0.65:  return "Moderate"
    return "High"


def _compute_shap(x_scaled: np.ndarray, x_raw: np.ndarray) -> list[SHAPValue]:
    """Compute per-feature SHAP contributions for a single sample."""
    try:
        sv = _explainer.shap_values(x_scaled)
        # sv can be list[array] for binary classifiers
        if isinstance(sv, list):
            impacts = np.array(sv[1][0])
        else:
            arr = np.array(sv)
            if arr.ndim == 3:       # (1, features, classes)
                impacts = arr[0, :, 1]
            elif arr.ndim == 2:     # (1, features)
                impacts = arr[0]
            else:
                impacts = arr

        result = []
        for i, fname in enumerate(FEATURE_NAMES):
            result.append(SHAPValue(
                feature=fname,
                value=float(x_raw[0, i]),
                impact=round(float(impacts[i]), 4),
            ))
        return sorted(result, key=lambda s: abs(s.impact), reverse=True)

    except Exception as e:
        # Graceful degradation — return zeros so the API still responds
        print(f"⚠️  SHAP computation failed: {e}")
        return [SHAPValue(feature=f, value=float(x_raw[0, i]), impact=0.0)
                for i, f in enumerate(FEATURE_NAMES)]


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Liveness check."""
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/meta", tags=["Info"])
def get_meta():
    """
    Returns feature metadata (labels, ranges, options) and model evaluation
    metrics for all trained candidates.
    """
    return _meta


@app.get("/feature-importance", tags=["Info"])
def feature_importance():
    """Global SHAP-based feature importance (mean |SHAP|)."""
    return {"feature_importance": _meta.get("feature_importance", [])}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(patient: PatientInput):
    """
    Predict heart disease risk from patient clinical data.

    Returns:
    - **prediction**: 0 (No Disease) or 1 (Disease)
    - **probability**: probability of heart disease (0–1)
    - **risk_level**: Low / Moderate / High
    - **shap_values**: per-feature SHAP contributions, sorted by impact
    """
    t0 = time.perf_counter()

    # Build feature vector in correct column order
    x_raw = np.array([[
        patient.age, patient.sex, patient.cp, patient.trestbps, patient.chol,
        patient.fbs, patient.restecg, patient.thalach, patient.exang,
        patient.oldpeak, patient.slope, patient.ca, patient.thal
    ]], dtype=float)

    x_scaled = _scaler.transform(x_raw)

    proba      = _model.predict_proba(x_scaled)[0]   # [P(0), P(1)]
    prediction = int(np.argmax(proba))
    prob_disease = float(proba[1])

    shap_vals = _compute_shap(x_scaled, x_raw)

    elapsed = (time.perf_counter() - t0) * 1000

    return PredictionResponse(
        prediction=prediction,
        probability=round(prob_disease, 4),
        risk_level=_risk_level(prob_disease),
        confidence=round(float(max(proba)), 4),
        shap_values=shap_vals,
        elapsed_ms=round(elapsed, 2),
    )
