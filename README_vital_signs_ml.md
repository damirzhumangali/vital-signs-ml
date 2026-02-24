# Vital Signs Anomaly Detection — ML Pipeline

A machine learning pipeline for detecting anomalies in patient vital signs — extending the IoT-based [Patient Health Monitoring System](https://github.com/damirzhumangali/Patient_Health_Monitoring) with intelligent data analysis.

**Dataset:** UCI Heart Disease Dataset (303 patients, 14 features)  
**Task:** Binary classification + unsupervised anomaly detection

---

## Motivation

The Patient Health Monitoring System collects real-time vital sign data (heart rate, SpO2, body temperature) from ESP32 sensors. Threshold-based alerts — "flag if BPM > 100" — are too brittle for clinical use: they don't adapt to patient-specific baselines, miss gradual deterioration, and generate excessive false alarms.

This project builds the ML layer: models that *learn* what normal looks like for a given patient population, and detect deviations that thresholds would miss.

The core research question mirrors Dr. Ayhan's work on clinical deep learning validation: **how do we build ML models that remain trustworthy when input data is noisy, heterogeneous, and arrives from constrained hardware?**

---

## Pipeline

```
Raw biomedical data (UCI Heart Disease / ESP32 sensor streams)
        ↓
1. Exploratory Data Analysis (Pandas, Matplotlib, Seaborn)
        ↓
2. Preprocessing (imputation, scaling, stratified split)
        ↓
3. Classical ML comparison (Logistic Regression, Random Forest, GBM, SVM)
        ↓
4. Unsupervised Anomaly Detection (Isolation Forest)
        ↓
5. Deep Learning time-series model (PyTorch LSTM)
        ↓
6. Clinical validation (ROC-AUC, Sensitivity, Specificity, Precision-Recall)
```

---

## Key Results

| Model | ROC-AUC | Sensitivity | Specificity |
|---|---|---|---|
| Logistic Regression | — | — | — |
| Random Forest | — | — | — |
| Gradient Boosting | — | — | — |
| SVM (RBF) | — | — | — |
| LSTM (PyTorch) | — | — | — |

*Run the notebook to populate results.*

---

## Clinical Validation Insight

Standard accuracy is insufficient for clinical AI. In medical settings, **sensitivity** (true positive rate — not missing a disease case) is weighted differently than specificity — a missed diagnosis is more costly than a false alarm.

This tradeoff is explicit in the evaluation: all models are assessed on sensitivity, specificity, ROC-AUC, and precision-recall curves, not just accuracy.

---

## Unsupervised Anomaly Detection

In real IoT deployments, labeled anomalies are rare — sensors stream continuously without ground truth labels. The Isolation Forest model is trained *only* on healthy patient data, then identifies statistically isolated points as anomalies. Any performance above random baseline demonstrates meaningful detection without supervision.

---

## Tech Stack

| Purpose | Library |
|---|---|
| Data manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Classical ML | scikit-learn |
| Anomaly detection | scikit-learn (Isolation Forest) |
| Deep learning | PyTorch |
| Dataset | ucimlrepo |

---

## Setup

```bash
git clone https://github.com/damirzhumangali/vital-signs-ml.git
cd vital-signs-ml
pip install pandas numpy matplotlib seaborn scikit-learn torch ucimlrepo
jupyter notebook vital_signs_ml_pipeline.ipynb
```

---

## Connection to IoT Pipeline

This project is the analytical layer on top of the [Patient Health Monitoring System](https://github.com/damirzhumangali/Patient_Health_Monitoring):

```
ESP32 sensors → MQTT → data stream → [THIS PROJECT] → anomaly alerts
```

**Next steps:**
- Replace UCI dataset with real ESP32 sensor readings
- Per-patient personalized baseline models
- Edge inference deployment on Raspberry Pi 4 (no cloud dependency)
- Federated learning across devices without centralizing patient data

---

## Author

**Damir Zhumangali** — [github.com/damirzhumangali](https://github.com/damirzhumangali)
