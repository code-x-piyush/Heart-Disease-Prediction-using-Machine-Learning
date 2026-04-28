# 🫀 CardioScope — AI-Powered Heart Disease Prediction

A full-stack, production-ready web application for cardiovascular risk assessment using machine learning with SHAP explainability.

---

## ✨ Features

| Layer | Stack |
|---|---|
| **Frontend** | React 18 + Vite · Custom CSS (no Tailwind dependency) |
| **Backend** | Python 3.11 · FastAPI · Uvicorn |
| **ML** | scikit-learn (Logistic Regression, Random Forest, Gradient Boosting) |
| **Explainability** | SHAP (KernelExplainer / TreeExplainer auto-selected) |
| **Deployment** | Backend → Render · Frontend → Vercel · Docker Compose |

---

## 📁 Project Structure

```
heart-disease-app/
├── backend/
│   ├── main.py              # FastAPI application & endpoints
│   ├── train_model.py       # ML pipeline (data → model artifacts)
│   ├── model/               # Generated artifacts (after training)
│   │   ├── model.pkl        # Best trained model
│   │   ├── scaler.pkl       # StandardScaler
│   │   └── meta.json        # Feature metadata + evaluation metrics
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main React application
│   │   └── main.jsx         # Entry point
│   ├── public/
│   │   └── heart.svg        # Favicon
│   ├── index.html
│   ├── vite.config.js
│   ├── package.json
│   ├── .env.example
│   ├── vercel.json
│   └── Dockerfile
│
├── docker-compose.yml        # Full-stack Docker orchestration
├── render.yaml               # Render.com deployment blueprint
└── README.md
```

---

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.11+
- Node.js 20+
- pip

---

### 1. Backend Setup

```bash
cd backend

# Create virtual environment
.\.venv\Scripts\Activate.ps1        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model (generates model/ artifacts)
python train_model.py
# Expected output:
#   ✅ Loaded dataset from UCI (303 rows)   ← or synthetic fallback
#   🏆 Best model: Logistic Regression (ROC-AUC = 0.91)
#   💾 Artifacts saved to ./model/

# Start the API server
uvicorn main:app --reload --port 8000
```

API is now running at **http://localhost:8000**

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure API URL (optional – defaults to localhost:8000)
cp .env.example .env.local
# edit VITE_API_URL if needed

# Start dev server
npm run dev
```

Frontend is now running at **http://localhost:3000**

---

## 🐳 Docker (Full Stack)

```bash
# Build and start both services
docker-compose up --build

# Frontend → http://localhost:3000
# Backend  → http://localhost:8000
```

The Docker build automatically runs `train_model.py` during the backend image build, so no manual training step is needed.

---

## 🌐 Deployment

### Backend → Render

1. Push the repo to GitHub
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your repo and point to the `backend/` directory, OR use the included `render.yaml` blueprint:
   ```bash
   # From repo root, Render detects render.yaml automatically
   ```
4. Render will run `python train_model.py` then `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Copy the Render URL (e.g. `https://cardioscope-api.onrender.com`)

### Frontend → Vercel

```bash
cd frontend

# Install Vercel CLI
npm i -g vercel

# Deploy (first time: follow prompts)
vercel

# Set environment variable in Vercel dashboard:
# VITE_API_URL = https://cardioscope-api.onrender.com

# Production deploy
vercel --prod
```

Or via Vercel Dashboard:
1. Import the GitHub repo → set **Root Directory** to `frontend`
2. Add **Environment Variable**: `VITE_API_URL` = your Render URL
3. Deploy

---

## 📡 API Reference

### `POST /predict`

Predict heart disease risk from patient data.

**Request body:**
```json
{
  "age": 54,
  "sex": 1,
  "cp": 0,
  "trestbps": 122,
  "chol": 286,
  "fbs": 0,
  "restecg": 0,
  "thalach": 116,
  "exang": 1,
  "oldpeak": 3.2,
  "slope": 1,
  "ca": 2,
  "thal": 2
}
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.9948,
  "risk_level": "High",
  "confidence": 0.9948,
  "shap_values": [
    { "feature": "ca", "value": 2.0, "impact": 0.312 },
    { "feature": "oldpeak", "value": 3.2, "impact": 0.289 }
  ],
  "elapsed_ms": 42.5
}
```

### `GET /meta`
Returns feature labels, ranges, options, and model evaluation metrics.

### `GET /feature-importance`
Returns global SHAP-based feature importance ranking.

### `GET /health`
Liveness check — `{"status": "ok", "model_loaded": true}`.

---

## 🤖 Machine Learning Pipeline

### Dataset
UCI Cleveland Heart Disease dataset (303 samples, 13 features).
The training script attempts to fetch from the UCI repository; if unavailable, a statistically calibrated synthetic replica is used automatically.

### Features

| Feature | Description |
|---|---|
| age | Age in years |
| sex | 0 = Female, 1 = Male |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved (bpm) |
| exang | Exercise-induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels colored by fluoroscopy |
| thal | Thalassemia type |

### Model Selection

Three models are trained and evaluated. The best by ROC-AUC is automatically selected:

| Model | Typical ROC-AUC |
|---|---|
| Logistic Regression | ~0.91 |
| Random Forest | ~0.86 |
| Gradient Boosting | ~0.83 |

### Preprocessing
- Missing values imputed with column median
- Outliers clipped at 1st / 99th percentile
- Features standardized with `StandardScaler`
- Target binarized: 0 = No Disease, 1 = Disease

### Explainability
SHAP values are computed per prediction:
- **TreeExplainer** for tree-based models (fast)
- **KernelExplainer** for linear models (approximate)
- Each feature's contribution to the prediction is returned in the API response, sorted by absolute impact

---

## 🏥 Medical Disclaimer

> This application is intended for **research and educational purposes only**.
> It is **not a medical device** and must not be used for clinical diagnosis or treatment decisions.
> Always consult a qualified healthcare professional.

---

## 📄 License

MIT License — free to use, modify, and distribute.
