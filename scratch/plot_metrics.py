import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create artifacts directory if it doesn't exist
os.makedirs('artifacts', exist_ok=True)

# Set style
sns.set_theme(style="whitegrid")

# 1. Confusion Matrix
plt.figure(figsize=(6, 5))
cm = np.array([[96207, 19], [1, 3773]])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Attack'], 
            yticklabels=['Normal', 'Attack'],
            annot_kws={"size": 14})
plt.title('Confusion Matrix', fontsize=16)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('artifacts/confusion_matrix.png', dpi=300)
plt.close()

# 2. Classification Metrics Bar Chart
plt.figure(figsize=(8, 5))
metrics = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
values = [0.994989, 0.999735, 0.997357, 0.999998]
colors = ['#388bfd', '#3fb950', '#d29922', '#f85149']

ax = sns.barplot(x=metrics, y=values, palette=colors)
plt.title('Classification Metrics', fontsize=16)
plt.ylim(0.99, 1.001)  # Zoom in to see differences
for i, v in enumerate(values):
    ax.text(i, v + 0.0005, f'{v:.4f}', ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('artifacts/classification_metrics.png', dpi=300)
plt.close()

# 3. Streaming Performance (Latency)
plt.figure(figsize=(8, 5))
latency_metrics = ['Mean', 'Median', 'p95', 'p99']
latency_values = [4.2256, 4.0914, 4.5980, 4.9838]

ax = sns.barplot(x=latency_metrics, y=latency_values, color='#8b949e')
plt.title('Inference Latency Metrics (ms)', fontsize=16)
plt.ylabel('Latency (ms)')
for i, v in enumerate(latency_values):
    ax.text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('artifacts/latency_metrics.png', dpi=300)
plt.close()

print("Visuals generated successfully in artifacts/ directory.")
