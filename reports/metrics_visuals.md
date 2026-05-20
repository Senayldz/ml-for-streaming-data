# SWaT Streaming Anomaly Detection Metrics (Temporal Split & Isolation Forest)

Based on the evaluation pipeline using **Isolation Forest** with a chronological **Temporal Split (80/20)**, here are the detailed performance and latency metrics.

> [!NOTE]
> The model was trained **exclusively on Normal data** (first 80% of the dataset, 1,153,375 records), with zero exposure to siber attacks during training. The evaluation was run on the remaining 20% test split (288,344 records) which contains an attack ratio of 18.94%.

## 1. Classification Metrics

The model achieves strong, realistic generalization performance without any data leakage:

| Metric | Score | Note |
|---|---|---|
| **Precision** | 0.9576 | Extremely low false alarms, high confidence alerts |
| **Recall** | 0.6899 | Detects ~69% of attack records |
| **F1-Score** | 0.8020 | Strong harmonic balance for unsupervised detection |
| **ROC-AUC** | 0.9431 | Outlier rank score |

## 2. Confusion Matrix

The confusion matrix breakdown on 288,344 evaluated records:

| Label | True Normal | True Attack |
|---|---|---|
| **Predicted Normal** | 232,053 (TN) | 16,937 (FN) |
| **Predicted Attack** | 1,670 (FP) | 37,684 (TP) |

> [!IMPORTANT]
> The **False Positives (FP) is 1,670**, representing a **False Alarm Rate of only 0.71%**. In a real-world control room, this keeps operator alert fatigue extremely low while still flagging 37,684 attack records.

## 3. Streaming Inference Performance (Latency)

The unsupervised Isolation Forest model processes batches efficiently:

- **Throughput:** ~3,727 records per second (14.57 batches of 256 records/sec)
- **Mean Latency:** 68.42 ms (per batch of 256)
- **95th Percentile (p95):** 74.87 ms
- **99th Percentile (p99):** 83.24 ms

> [!TIP]
> A sub-70ms latency per batch allows real-time inference on streaming industrial IoT data without causing any backpressure.

---

## 4. Concept Drift Scenario

In long-running industrial environments like SWaT, normal operational behavior slowly changes over time due to sensor degradation or mechanical wear and tear. This is known as **Concept Drift**. If the model is not retrained, this gradual shift will cause an increase in False Positives (alert fatigue).

Below is a visualization illustrating how unmitigated concept drift impacts model performance until a retraining cycle (Active Learning) resets it.

![Concept Drift Simulation](concept_drift.png)
