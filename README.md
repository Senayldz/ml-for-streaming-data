# 🛡️ SWaT Streaming Anomaly Detection & Security Radar

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ml-for-streaming-data-ttzdt5cixfpqq6gkdedse7.streamlit.app/)

A real-time, high-performance streaming machine learning pipeline and interactive dashboard for cyber-physical attack detection on the **Secure Water Treatment (SWaT)** industrial dataset. Built with unsupervised Isolation Forest, Streamlit, and Plotly, this system features sub-70ms batch inference latency, thread-safe background streaming simulation, and a fully responsive user interface.

---

## 🚀 Key Features

* **Incremental Streaming Pipeline:** Simulates live PLC sensor/actuator data ingestion with mini-batch generation.
* **Unsupervised Anomaly Detector:** Isolation Forest baseline trained exclusively on normal operations, achieving a highly realistic **95.8% Precision**, **69.0% Recall**, and **94.3% ROC-AUC** under a strict temporal split.
* **Real-time Interactive Dashboard:** 
  * Live-updating Threat Score charts (Plotly).
  * Attack injection simulator (Valve, Pump, Tank attacks) to test detector responsiveness.
  * AI-powered security assistant commentary reflecting live conditions.
  * Tech logs displaying anomaly classifications and full streams.
* **Responsive Styling:** Optimized layout that adapts to Mobile, Tablet, and Desktop screens.
* **Automated Evaluation Report:** Generates static plots, a detailed report text file, and a formatted PowerPoint evaluation slide deck.

---

## 📊 Performance & Evaluation Results

The pipeline was validated using a held-out 20% chronological test split from `merged.csv`, evaluating the full **288,344 records** on standard CPU hardware.

### 1. Classification Metrics
The unsupervised model achieves highly realistic, robust separation without any time-series data leakage:

| Metric | Score | Detail |
|---|---|---|
| **Precision** | **95.76%** | Extremely low false alarm rate (0.71%), avoiding operator fatigue |
| **Recall** | **68.99%** | Catches 69% of cyber-physical attacks with zero prior exposure to attack patterns |
| **F1-Score** | **80.20%** | Strong overall harmonic balance for an unsupervised detector |
| **ROC-AUC** | **94.31%** | Strong global class separability based on anomaly scores |

> [!NOTE]
> **Why are these results realistic compared to 99.9% metrics?**
> Supervised models trained on **randomly shuffled splits** suffer from severe time-series **Data Leakage**. Shuffling puts adjacent seconds of the same attack sequences into both train and test splits, causing the model to memorize the specific signatures.
> By switching to a **chronological Temporal Split (80/20)**, the model is trained exclusively on normal operation data and tested on subsequent unseen timeline blocks. The unsupervised Isolation Forest learns what normal looks like, resulting in a robust, generalizable model for actual production environments.

### 2. Confusion Matrix
On the 288,344 test records, the model achieves a low False Alarm Rate of **0.71%**:

| Actual \ Predicted | Predicted Normal | Predicted Attack |
|---|---|---|
| **True Normal** | **232,053** (True Negative) | **1,670** (False Positive) |
| **True Attack** | **16,937** (False Negative) | **37,684** (True Positive) |

### 3. Real-Time Streaming Latency
The inference engine is designed for real-time edge applications:

* **Throughput:** ~3,727 records / second (batch=256).
* **Mean Batch Latency (batch=256):** **68.42 ms**
* **p95 Latency:** **74.87 ms**
* **p99 Latency:** **83.24 ms**

---

## 🔄 Concept Drift Scenario

In industrial environments, physical wear, sensor degradation, and mechanical changes introduce **Concept Drift**. If the model is not retrained regularly, drift leads to a degradation of precision and an increase in false alarms. Below is the visualization of the simulation showing drift impact and model recovery after active retraining.

![Concept Drift Simulation](reports/concept_drift.png)

---

## 🛠️ Setup & Installation

### Prerequisites
* Python 3.7+
* Pip

### 1. Clone the Repository & Install Dependencies
```bash
git clone https://github.com/Senayldz/ml-for-streaming-data.git
cd ml-for-streaming-data
pip install -r requirements.txt
```

### 2. Run the Interactive Dashboard
Launch the web interface locally:
```bash
streamlit run app.py
```
* Access the interface in your browser at: `http://localhost:8501`

### 3. Run Pipeline CLI Tasks
You can run different CLI tasks using flags:
* **Train and Evaluate pipeline:**
  ```bash
  python app.py --model lgbm
  ```
* **Downsize full dataset for dashboard:**
  ```bash
  python app.py --create-mini
  ```
* **Generate PPTX Evaluation presentation:**
  ```bash
  python app.py --generate-presentation
  ```
