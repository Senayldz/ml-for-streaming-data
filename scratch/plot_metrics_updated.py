import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create reports directory if it doesn't exist
os.makedirs('reports', exist_ok=True)

# Set style
sns.set_theme(style="whitegrid")

# 1. Confusion Matrix
plt.figure(figsize=(6, 5))
cm = np.array([[232053, 1670], [16937, 37684]])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Attack'], 
            yticklabels=['Normal', 'Attack'],
            annot_kws={"size": 14})
plt.title('Confusion Matrix\n(Unsupervised Isolation Forest)', fontsize=16, fontweight='bold', pad=15)
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/confusion_matrix.png', dpi=300)
plt.close()

# 2. Classification Metrics Bar Chart
plt.figure(figsize=(8, 5))
metrics = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
values = [0.9576, 0.6899, 0.8020, 0.9431]
colors = ['#388bfd', '#3fb950', '#d29922', '#f85149']

ax = sns.barplot(x=metrics, y=values, palette=colors)
plt.title('Classification Metrics\n(Unsupervised Isolation Forest)', fontsize=16, fontweight='bold', pad=15)
plt.ylim(0.0, 1.1)
for i, v in enumerate(values):
    ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontsize=12, fontweight='bold')
plt.ylabel('Score', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/classification_metrics.png', dpi=300)
plt.close()

# 3. Streaming Performance (Latency)
plt.figure(figsize=(8, 5))
latency_metrics = ['Mean', 'Median', 'p95', 'p99']
latency_values = [68.42, 67.83, 74.87, 83.24]

ax = sns.barplot(x=latency_metrics, y=latency_values, color='#8b949e')
plt.title('Streaming Inference Latency per Batch (ms)\n(Batch Size = 256)', fontsize=16, fontweight='bold', pad=15)
plt.ylabel('Latency (ms)', fontsize=12, fontweight='bold')
for i, v in enumerate(latency_values):
    ax.text(i, v + 1.5, f'{v:.2f} ms', ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/latency_metrics.png', dpi=300)
plt.close()

print("All updated visuals generated successfully in reports/ directory.")
