import { useState, useEffect, useRef } from "react";

// ─── Constants ────────────────────────────────────────────────────────────────

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

const FEATURE_META = {
  age:      { label: "Age", unit: "years", min: 20, max: 80, step: 1, type: "number", default: 54 },
  sex:      { label: "Sex", type: "select", default: 1,
               options: [{ value: 0, label: "Female" }, { value: 1, label: "Male" }] },
  cp:       { label: "Chest Pain Type", type: "select", default: 0,
               options: [
                 { value: 0, label: "Typical Angina" },
                 { value: 1, label: "Atypical Angina" },
                 { value: 2, label: "Non-anginal Pain" },
                 { value: 3, label: "Asymptomatic" },
               ]},
  trestbps: { label: "Resting Blood Pressure", unit: "mm Hg", min: 80, max: 200, step: 1, type: "number", default: 122 },
  chol:     { label: "Serum Cholesterol", unit: "mg/dl", min: 100, max: 600, step: 1, type: "number", default: 240 },
  fbs:      { label: "Fasting Blood Sugar > 120 mg/dl", type: "select", default: 0,
               options: [{ value: 0, label: "No (≤120)" }, { value: 1, label: "Yes (>120)" }] },
  restecg:  { label: "Resting ECG Results", type: "select", default: 0,
               options: [
                 { value: 0, label: "Normal" },
                 { value: 1, label: "ST-T Abnormality" },
                 { value: 2, label: "LV Hypertrophy" },
               ]},
  thalach:  { label: "Max Heart Rate", unit: "bpm", min: 60, max: 220, step: 1, type: "number", default: 150 },
  exang:    { label: "Exercise Induced Angina", type: "select", default: 0,
               options: [{ value: 0, label: "No" }, { value: 1, label: "Yes" }] },
  oldpeak:  { label: "ST Depression (Exercise)", unit: "mm", min: 0, max: 6.2, step: 0.1, type: "number", default: 1.0 },
  slope:    { label: "Slope of Peak ST Segment", type: "select", default: 1,
               options: [
                 { value: 0, label: "Upsloping" },
                 { value: 1, label: "Flat" },
                 { value: 2, label: "Downsloping" },
               ]},
  ca:       { label: "Major Vessels Colored", type: "select", default: 0,
               options: [
                 { value: 0, label: "0 vessels" },
                 { value: 1, label: "1 vessel" },
                 { value: 2, label: "2 vessels" },
                 { value: 3, label: "3 vessels" },
               ]},
  thal:     { label: "Thalassemia", type: "select", default: 2,
               options: [
                 { value: 1, label: "Normal" },
                 { value: 2, label: "Fixed Defect" },
                 { value: 3, label: "Reversible Defect" },
               ]},
};

const FEATURE_KEYS = Object.keys(FEATURE_META);

const initialForm = () =>
  Object.fromEntries(FEATURE_KEYS.map((k) => [k, FEATURE_META[k].default]));

// ─── Helpers ─────────────────────────────────────────────────────────────────

function riskColor(level) {
  if (level === "Low")      return { bg: "#0d9488", text: "#ccfbf1", bar: "#14b8a6" };
  if (level === "Moderate") return { bg: "#d97706", text: "#fef3c7", bar: "#f59e0b" };
  return                           { bg: "#dc2626", text: "#fee2e2", bar: "#ef4444" };
}

function riskEmoji(level) {
  if (level === "Low")      return "💚";
  if (level === "Moderate") return "🟡";
  return "🔴";
}

function formatDate(iso) {
  return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function HeartPulse({ active }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
        <path
          d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"
          fill={active ? "#ef4444" : "#374151"}
          style={{ transition: "fill 0.4s" }}
        >
          {active && (
            <animate attributeName="opacity" values="1;0.6;1" dur="1s" repeatCount="indefinite" />
          )}
        </path>
      </svg>
      <span style={{ fontFamily: "'Playfair Display', serif", fontSize: 22, fontWeight: 700, color: "#f1f5f9", letterSpacing: "-0.5px" }}>
        CardioScope
      </span>
    </div>
  );
}

function StatCard({ label, value, sub, color = "#6366f1" }) {
  return (
    <div style={{
      background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)",
      borderRadius: 12, padding: "16px 20px", flex: 1, minWidth: 120,
    }}>
      <div style={{ fontSize: 11, color: "#94a3b8", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 26, fontWeight: 800, color, fontFamily: "'DM Mono', monospace" }}>{value}</div>
      {sub && <div style={{ fontSize: 12, color: "#64748b", marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

function AnimatedBar({ value, color, delay = 0 }) {
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setWidth(value * 100), delay + 80);
    return () => clearTimeout(t);
  }, [value, delay]);

  return (
    <div style={{ background: "rgba(255,255,255,0.07)", borderRadius: 6, height: 8, overflow: "hidden" }}>
      <div style={{
        width: `${width}%`, height: "100%", borderRadius: 6,
        background: `linear-gradient(90deg, ${color}, ${color}cc)`,
        transition: "width 0.7s cubic-bezier(0.4,0,0.2,1)",
        boxShadow: `0 0 8px ${color}88`,
      }} />
    </div>
  );
}

function ShapBar({ name, impact, maxImpact }) {
  const pct = Math.abs(impact) / (maxImpact || 1);
  const positive = impact > 0;
  const color = positive ? "#ef4444" : "#22c55e";
  const label = FEATURE_META[name]?.label || name;

  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4, fontSize: 12 }}>
        <span style={{ color: "#cbd5e1" }}>{label}</span>
        <span style={{ color, fontFamily: "'DM Mono', monospace", fontWeight: 600 }}>
          {positive ? "+" : ""}{impact.toFixed(3)}
        </span>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <div style={{ flex: 1, background: "rgba(255,255,255,0.05)", borderRadius: 4, height: 6, position: "relative" }}>
          <div style={{
            position: "absolute",
            [positive ? "left" : "right"]: "50%",
            width: `${pct * 50}%`,
            height: "100%",
            background: color,
            borderRadius: 4,
            transition: "width 0.6s ease",
            boxShadow: `0 0 6px ${color}88`,
          }} />
          <div style={{ position: "absolute", left: "50%", top: 0, width: 1, height: "100%", background: "rgba(255,255,255,0.2)" }} />
        </div>
      </div>
    </div>
  );
}

function HistoryItem({ item, onClick, isSelected }) {
  const colors = riskColor(item.risk_level);
  return (
    <div
      onClick={() => onClick(item)}
      style={{
        padding: "10px 14px", borderRadius: 10, cursor: "pointer", marginBottom: 6,
        background: isSelected ? "rgba(99,102,241,0.15)" : "rgba(255,255,255,0.03)",
        border: `1px solid ${isSelected ? "#6366f1" : "rgba(255,255,255,0.06)"}`,
        display: "flex", alignItems: "center", gap: 12, transition: "all 0.2s",
      }}
    >
      <div style={{
        width: 36, height: 36, borderRadius: 8,
        background: colors.bg + "22",
        border: `1.5px solid ${colors.bar}`,
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: 16,
      }}>{riskEmoji(item.risk_level)}</div>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 13, color: "#e2e8f0", fontWeight: 600 }}>
          {Math.round(item.probability * 100)}% risk · {item.risk_level}
        </div>
        <div style={{ fontSize: 11, color: "#64748b" }}>{formatDate(item.timestamp)}</div>
      </div>
    </div>
  );
}

function FormField({ name, value, onChange }) {
  const meta = FEATURE_META[name];
  const inputStyle = {
    width: "100%", padding: "9px 12px", borderRadius: 8, border: "1px solid rgba(255,255,255,0.12)",
    background: "rgba(255,255,255,0.05)", color: "#f1f5f9", fontSize: 13,
    outline: "none", boxSizing: "border-box",
    transition: "border-color 0.2s",
  };

  return (
    <div style={{ marginBottom: 4 }}>
      <label style={{ display: "block", fontSize: 11, color: "#94a3b8", marginBottom: 5, fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.06em" }}>
        {meta.label}{meta.unit ? <span style={{ color: "#475569", marginLeft: 4 }}>({meta.unit})</span> : ""}
      </label>
      {meta.type === "select" ? (
        <select value={value} onChange={(e) => onChange(name, Number(e.target.value))}
          style={{ ...inputStyle, cursor: "pointer" }}>
          {meta.options.map((o) => (
            <option key={o.value} value={o.value} style={{ background: "#1e293b" }}>{o.label}</option>
          ))}
        </select>
      ) : (
        <input
          type="number" value={value} min={meta.min} max={meta.max} step={meta.step || 1}
          onChange={(e) => onChange(name, parseFloat(e.target.value) || 0)}
          style={inputStyle}
        />
      )}
    </div>
  );
}

// ─── Result Panel ─────────────────────────────────────────────────────────────

function ResultPanel({ result }) {
  const colors = riskColor(result.risk_level);
  const pct = Math.round(result.probability * 100);
  const maxImpact = Math.max(...result.shap_values.map((s) => Math.abs(s.impact)));
  const [showAll, setShowAll] = useState(false);
  const shaps = showAll ? result.shap_values : result.shap_values.slice(0, 6);

  return (
    <div style={{ animation: "fadeSlideUp 0.4s ease" }}>
      {/* Risk banner */}
      <div style={{
        borderRadius: 14, padding: "24px 28px", marginBottom: 20,
        background: `linear-gradient(135deg, ${colors.bg}33, ${colors.bg}11)`,
        border: `1.5px solid ${colors.bar}55`,
        display: "flex", alignItems: "center", gap: 20,
      }}>
        <div style={{
          width: 72, height: 72, borderRadius: "50%",
          background: `conic-gradient(${colors.bar} ${pct * 3.6}deg, rgba(255,255,255,0.08) 0deg)`,
          display: "flex", alignItems: "center", justifyContent: "center",
          boxShadow: `0 0 20px ${colors.bar}44`,
          position: "relative",
        }}>
          <div style={{
            width: 54, height: 54, borderRadius: "50%",
            background: "#0f172a", display: "flex", alignItems: "center", justifyContent: "center",
            flexDirection: "column",
          }}>
            <span style={{ fontSize: 18, fontWeight: 900, color: colors.bar, fontFamily: "'DM Mono', monospace" }}>{pct}%</span>
          </div>
        </div>
        <div>
          <div style={{ fontSize: 13, color: "#94a3b8", marginBottom: 4 }}>Heart Disease Risk</div>
          <div style={{ fontSize: 28, fontWeight: 800, color: colors.bar, lineHeight: 1 }}>
            {result.risk_level} Risk
          </div>
          <div style={{ fontSize: 12, color: "#64748b", marginTop: 6 }}>
            {result.prediction === 1
              ? "Positive indication — consult a cardiologist"
              : "Negative indication — maintain healthy lifestyle"}
          </div>
        </div>
      </div>

      {/* Stats row */}
      <div style={{ display: "flex", gap: 10, marginBottom: 20, flexWrap: "wrap" }}>
        <StatCard label="Probability" value={`${pct}%`} color={colors.bar} />
        <StatCard label="Confidence" value={`${Math.round(result.confidence * 100)}%`} color="#6366f1" />
        <StatCard label="Latency" value={`${result.elapsed_ms}ms`} color="#22d3ee" />
      </div>

      {/* Progress bar */}
      <div style={{ marginBottom: 20 }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#64748b", marginBottom: 6 }}>
          <span>Risk Score</span><span>{pct}%</span>
        </div>
        <AnimatedBar value={result.probability} color={colors.bar} />
      </div>

      {/* SHAP values */}
      <div style={{
        background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)",
        borderRadius: 12, padding: "16px 18px",
      }}>
        <div style={{ fontSize: 12, color: "#94a3b8", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.07em", marginBottom: 14 }}>
          Feature Contributions (SHAP)
        </div>
        <div style={{ fontSize: 11, color: "#475569", marginBottom: 12, display: "flex", gap: 16 }}>
          <span><span style={{ color: "#ef4444" }}>■</span> Increases risk</span>
          <span><span style={{ color: "#22c55e" }}>■</span> Decreases risk</span>
        </div>
        {shaps.map((s) => (
          <ShapBar key={s.feature} name={s.feature} impact={s.impact} maxImpact={maxImpact} />
        ))}
        {result.shap_values.length > 6 && (
          <button
            onClick={() => setShowAll(!showAll)}
            style={{
              marginTop: 8, fontSize: 12, color: "#6366f1", background: "none", border: "none",
              cursor: "pointer", padding: 0, textDecoration: "underline",
            }}
          >
            {showAll ? "Show less" : `Show ${result.shap_values.length - 6} more`}
          </button>
        )}
      </div>
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [form, setForm]         = useState(initialForm());
  const [result, setResult]     = useState(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);
  const [history, setHistory]   = useState([]);
  const [activeTab, setActiveTab] = useState("form"); // "form" | "history" | "info"
  const [modelMeta, setModelMeta] = useState(null);
  const [selectedHist, setSelectedHist] = useState(null);

  // Fetch meta on mount
  useEffect(() => {
    fetch(`${API_BASE}/meta`)
      .then((r) => r.json())
      .then(setModelMeta)
      .catch(() => {});
  }, []);

  const handleChange = (key, val) => {
    setForm((f) => ({ ...f, [key]: val }));
    setResult(null);
    setError(null);
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const resp = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!resp.ok) {
        const err = await resp.json();
        throw new Error(err.detail || `HTTP ${resp.status}`);
      }
      const data = await resp.json();
      setResult(data);
      const entry = { ...data, timestamp: new Date().toISOString(), inputs: { ...form } };
      setHistory((h) => [entry, ...h].slice(0, 20));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const loadHistory = (item) => {
    setForm({ ...item.inputs });
    setResult(item);
    setSelectedHist(item.timestamp);
    setActiveTab("form");
  };

  // Group features into sections
  const sections = [
    { title: "Demographics", keys: ["age", "sex"] },
    { title: "Cardiac Symptoms", keys: ["cp", "exang", "slope"] },
    { title: "Vitals & Labs", keys: ["trestbps", "chol", "fbs", "thalach"] },
    { title: "Diagnostics", keys: ["restecg", "oldpeak", "ca", "thal"] },
  ];

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=DM+Sans:wght@400;500;600&family=DM+Mono:wght@400;500;600&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #080f1a; color: #e2e8f0; font-family: 'DM Sans', sans-serif; min-height: 100vh; }
        select option { background: #1e293b; color: #e2e8f0; }
        input[type=number]::-webkit-inner-spin-button { opacity: 0.3; }
        @keyframes fadeSlideUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes pulse-ring { 0%,100% { opacity: 0.6; } 50% { opacity: 1; } }
        @keyframes spin { to { transform: rotate(360deg); } }
        ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
      `}</style>

      {/* Background gradient mesh */}
      <div style={{
        position: "fixed", inset: 0, zIndex: 0, pointerEvents: "none",
        background: "radial-gradient(ellipse 80% 60% at 20% 0%, rgba(99,102,241,0.08) 0%, transparent 60%), radial-gradient(ellipse 60% 40% at 80% 100%, rgba(239,68,68,0.06) 0%, transparent 60%)",
      }} />

      <div style={{ position: "relative", zIndex: 1, minHeight: "100vh" }}>
        {/* ── Header ── */}
        <header style={{
          padding: "18px 28px", borderBottom: "1px solid rgba(255,255,255,0.06)",
          display: "flex", alignItems: "center", justifyContent: "space-between",
          backdropFilter: "blur(12px)", background: "rgba(8,15,26,0.8)",
          position: "sticky", top: 0, zIndex: 100,
        }}>
          <HeartPulse active={loading} />
          <div style={{ display: "flex", gap: 6 }}>
            {["form", "history", "info"].map((tab) => (
              <button key={tab} onClick={() => setActiveTab(tab)}
                style={{
                  padding: "7px 16px", borderRadius: 8, border: "none", cursor: "pointer",
                  fontSize: 13, fontWeight: 500, transition: "all 0.2s",
                  background: activeTab === tab ? "#6366f1" : "rgba(255,255,255,0.05)",
                  color: activeTab === tab ? "#fff" : "#94a3b8",
                }}>
                {tab === "form" ? "Predict" : tab === "history" ? `History${history.length ? ` (${history.length})` : ""}` : "Model Info"}
              </button>
            ))}
          </div>
        </header>

        {/* ── Main ── */}
        <main style={{ maxWidth: 1100, margin: "0 auto", padding: "28px 20px" }}>

          {/* ── PREDICT TAB ── */}
          {activeTab === "form" && (
            <div style={{ display: "grid", gridTemplateColumns: "minmax(0,1fr) minmax(0,420px)", gap: 24, alignItems: "start" }}>

              {/* Left: Form */}
              <div>
                <div style={{ marginBottom: 20 }}>
                  <h1 style={{ fontFamily: "'Playfair Display', serif", fontSize: 28, fontWeight: 800, color: "#f1f5f9", marginBottom: 6 }}>
                    Cardiovascular Risk Assessment
                  </h1>
                  <p style={{ color: "#64748b", fontSize: 14 }}>
                    Enter patient clinical data to predict heart disease probability using ensemble ML with SHAP explainability.
                  </p>
                </div>

                {sections.map((sec) => (
                  <div key={sec.title} style={{
                    background: "rgba(255,255,255,0.025)", border: "1px solid rgba(255,255,255,0.07)",
                    borderRadius: 14, padding: "18px 20px", marginBottom: 16,
                  }}>
                    <div style={{ fontSize: 12, color: "#6366f1", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 14 }}>
                      {sec.title}
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: 14 }}>
                      {sec.keys.map((k) => (
                        <FormField key={k} name={k} value={form[k]} onChange={handleChange} />
                      ))}
                    </div>
                  </div>
                ))}

                {/* Predict button */}
                <button onClick={handleSubmit} disabled={loading}
                  style={{
                    width: "100%", padding: "14px", borderRadius: 12, border: "none",
                    cursor: loading ? "not-allowed" : "pointer", fontSize: 15, fontWeight: 700,
                    background: loading ? "rgba(99,102,241,0.4)" : "linear-gradient(135deg, #6366f1, #8b5cf6)",
                    color: "#fff", letterSpacing: "0.02em", transition: "all 0.2s",
                    boxShadow: loading ? "none" : "0 4px 20px rgba(99,102,241,0.35)",
                    display: "flex", alignItems: "center", justifyContent: "center", gap: 10,
                  }}>
                  {loading ? (
                    <>
                      <div style={{
                        width: 18, height: 18, border: "2px solid rgba(255,255,255,0.3)",
                        borderTopColor: "#fff", borderRadius: "50%",
                        animation: "spin 0.8s linear infinite",
                      }} />
                      Analyzing…
                    </>
                  ) : "Analyze Patient Risk"}
                </button>

                {/* Error */}
                {error && (
                  <div style={{
                    marginTop: 14, padding: "12px 16px", borderRadius: 10,
                    background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)",
                    color: "#fca5a5", fontSize: 13,
                  }}>
                    ⚠️ {error}
                  </div>
                )}
              </div>

              {/* Right: Result */}
              <div style={{ position: "sticky", top: 80 }}>
                {result ? (
                  <ResultPanel result={result} />
                ) : (
                  <div style={{
                    borderRadius: 16, border: "1px dashed rgba(255,255,255,0.1)",
                    padding: "48px 24px", textAlign: "center",
                    background: "rgba(255,255,255,0.01)",
                  }}>
                    <div style={{ fontSize: 48, marginBottom: 16 }}>🫀</div>
                    <div style={{ color: "#475569", fontSize: 14, lineHeight: 1.6 }}>
                      Fill in patient data and click<br />
                      <strong style={{ color: "#6366f1" }}>Analyze Patient Risk</strong> to see<br />
                      the prediction and SHAP explanation.
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ── HISTORY TAB ── */}
          {activeTab === "history" && (
            <div>
              <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 24, fontWeight: 800, color: "#f1f5f9", marginBottom: 20 }}>
                Prediction History
              </h2>
              {history.length === 0 ? (
                <div style={{ textAlign: "center", padding: "60px 0", color: "#475569" }}>
                  No predictions yet. Go to Predict to get started.
                </div>
              ) : (
                <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: 24 }}>
                  <div>
                    {history.map((item) => (
                      <HistoryItem key={item.timestamp} item={item}
                        isSelected={selectedHist === item.timestamp}
                        onClick={loadHistory} />
                    ))}
                  </div>
                  {selectedHist && result && (
                    <div style={{
                      background: "rgba(255,255,255,0.025)", border: "1px solid rgba(255,255,255,0.07)",
                      borderRadius: 14, padding: 20,
                    }}>
                      <ResultPanel result={result} />
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* ── INFO TAB ── */}
          {activeTab === "info" && (
            <div>
              <h2 style={{ fontFamily: "'Playfair Display', serif", fontSize: 24, fontWeight: 800, color: "#f1f5f9", marginBottom: 20 }}>
                Model Performance
              </h2>
              {modelMeta ? (
                <>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 16, marginBottom: 28 }}>
                    {modelMeta.model_metrics?.map((m) => (
                      <div key={m.model} style={{
                        background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.08)",
                        borderRadius: 14, padding: "18px 20px",
                      }}>
                        <div style={{ fontSize: 14, fontWeight: 700, color: "#e2e8f0", marginBottom: 14 }}>{m.model}</div>
                        {["accuracy","precision","recall","f1","roc_auc"].map((k) => (
                          <div key={k} style={{ marginBottom: 10 }}>
                            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4 }}>
                              <span style={{ color: "#94a3b8", textTransform: "uppercase", letterSpacing: "0.05em" }}>{k.replace("_", " ")}</span>
                              <span style={{ color: "#e2e8f0", fontFamily: "'DM Mono', monospace" }}>{(m[k] * 100).toFixed(1)}%</span>
                            </div>
                            <AnimatedBar value={m[k]} color={k === "roc_auc" ? "#6366f1" : "#22d3ee"} />
                          </div>
                        ))}
                      </div>
                    ))}
                  </div>

                  {/* Global feature importance */}
                  <div style={{
                    background: "rgba(255,255,255,0.025)", border: "1px solid rgba(255,255,255,0.07)",
                    borderRadius: 14, padding: "20px 24px",
                  }}>
                    <div style={{ fontSize: 16, fontWeight: 700, color: "#e2e8f0", marginBottom: 18 }}>
                      Global Feature Importance (SHAP)
                    </div>
                    {modelMeta.feature_importance?.map((fi, idx) => (
                      <div key={fi.feature} style={{ marginBottom: 12 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 5 }}>
                          <span style={{ color: "#cbd5e1" }}>
                            #{idx + 1} {FEATURE_META[fi.feature]?.label || fi.feature}
                          </span>
                          <span style={{ color: "#6366f1", fontFamily: "'DM Mono', monospace" }}>{fi.importance.toFixed(4)}</span>
                        </div>
                        <AnimatedBar
                          value={fi.importance / (modelMeta.feature_importance[0]?.importance || 1)}
                          color="#6366f1"
                          delay={idx * 60}
                        />
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <div style={{ textAlign: "center", padding: "60px 0", color: "#475569" }}>
                  Could not load model metadata. Is the backend running?
                </div>
              )}
            </div>
          )}
        </main>

        {/* Footer */}
        <footer style={{ textAlign: "center", padding: "28px", borderTop: "1px solid rgba(255,255,255,0.05)", color: "#334155", fontSize: 12 }}>
          CardioScope · UCI Heart Disease Dataset · For research & educational use only
        </footer>
      </div>
    </>
  );
}
