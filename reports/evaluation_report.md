# SWaT Anomaly Detection — Evaluation Report

| Metric               | Value     | Note |
|----------------------|-----------|------|
| **Model**            | Isolation Forest | Unsupervised (Baseline) |
| **Split Strategy**   | Temporal (80% Train, 20% Test) | Realistic deployment scenario |
| **Precision**        | 0.957565  | High precision, extremely low false alarm rate |
| **Recall**           | 0.689918  | Successfully flags 69% of siber attack records |
| **F1-Score**         | 0.802001  | Strong overall balance for an unsupervised detector |
| **ROC-AUC**          | 0.943100  | Outlier score ranking metric |
| **True Positives**   | 37,684    | Attack records correctly identified |
| **True Negatives**   | 232,053   | Normal records correctly identified |
| **False Positives**  | 1,670     | Normal records falsely flagged as attacks (0.71% false alarm rate) |
| **False Negatives**  | 16,937    | Attack records missed by the model |
| **Records Streamed** | 288,344   | Full test split streamed |
| **Wall-clock Time**  | 77.37s    | Time taken for full test split streaming |
| **Throughput**       | 3,727.0   | Records processed per second |
| **Latency Mean**     | 68.42 ms  | Per batch of 256 records |
| **Latency Median**   | 67.83 ms  | Per batch of 256 records |
| **Latency p95**      | 74.87 ms  | Per batch of 256 records |
| **Latency p99**      | 83.24 ms  | Per batch of 256 records |
| **Latency Max**      | 95.00 ms  | Per batch of 256 records |
