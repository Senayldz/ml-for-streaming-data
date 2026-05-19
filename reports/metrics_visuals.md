# SWaT Streaming Anomaly Detection Metrics

Based on the evaluation pipeline using LightGBM, here are the detailed performance and latency metrics, along with their visualizations.

> [!NOTE]
> The evaluation was run on a held-out 20% stratified test split from `merged.csv`, processing 100,000 streaming records at 57,000 RPS.

## 1. Classification Metrics

The model achieves near-perfect performance on the test dataset:

| Metric | Score | Note |
|---|---|---|
| **Precision** | 0.9949 | Minimum false alarms |
| **Recall** | 0.9997 | Almost zero missed attacks |
| **F1-Score** | 0.9973 | Excellent harmonic balance |
| **ROC-AUC** | 0.9999 | Outstanding separability |

![Classification Metrics](classification_metrics.png)

## 2. Confusion Matrix

The confusion matrix breakdown on 100,000 evaluated records demonstrates the highly imbalanced nature of the dataset and how the model handled it successfully:

| Label | True Normal | True Attack |
|---|---|---|
| **Predicted Normal** | 96,207 (TN) | 1 (FN) |
| **Predicted Attack** | 19 (FP) | 3,773 (TP) |

![Confusion Matrix](confusion_matrix.png)

> [!IMPORTANT]
> The **False Negatives (FN) is 1**, meaning only **one attack record** slipped through undetected. This is critical for industrial security.

## 3. Streaming Inference Performance (Latency)

The model is highly optimized for real-time inference (on CPU only):

- **Throughput:** ~57,000 records per second
- **Mean Latency:** 4.22 ms (per batch of 256)
- **95th Percentile (p95):** 4.59 ms
- **99th Percentile (p99):** 4.98 ms

![Latency Metrics](latency_metrics.png)

> [!TIP]
> The sub-5ms mean latency makes this pipeline highly capable of handling real-time IoT scale ingestion from PLCs without backpressure.

---

## 4. Concept Drift Scenario

In long-running industrial environments like SWaT, normal operational behavior slowly changes over time due to sensor degradation or mechanical wear and tear. This is known as **Concept Drift**. If the model is not retrained, this gradual shift will cause an increase in False Positives (alert fatigue).

Below is a visualization illustrating how unmitigated concept drift impacts model performance until a retraining cycle (Active Learning) resets it.

![Concept Drift Simulation](concept_drift.png)
