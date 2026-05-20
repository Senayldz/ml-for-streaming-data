import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os

os.makedirs('reports', exist_ok=True)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.facecolor': '#0d1117',
    'figure.facecolor': '#0d1117',
    'text.color': '#c9d1d9',
    'axes.labelcolor': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'axes.edgecolor': '#30363d',
    'grid.color': '#21262d',
})

ACCENT_B = '#388bfd'
ACCENT_G = '#3fb950'
ACCENT_Y = '#d29922'
ACCENT_R = '#f85149'
WHITE = '#e6edf3'
SUBTEXT = '#8b949e'
BG_CARD = '#161b22'

# ── 1. Confusion Matrix (Dark Themed) ──
fig, ax = plt.subplots(figsize=(7, 5.5))
cm = np.array([[232053, 1670], [16937, 37684]])
labels = np.array([['TN\n232,053', 'FP\n1,670'], ['FN\n16,937', 'TP\n37,684']])

sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', linewidths=2, linecolor='#30363d',
            xticklabels=['Predicted: Normal', 'Predicted: Attack'],
            yticklabels=['Actual: Normal', 'Actual: Attack'],
            annot_kws={"size": 16, "weight": "bold", "color": WHITE},
            cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Confusion Matrix\nIsolation Forest — Temporal Split (80/20)', fontsize=15, fontweight='bold', color=WHITE, pad=15)
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('reports/confusion_matrix.png', dpi=200, bbox_inches='tight')
plt.close()
print('[1/6] confusion_matrix.png')

# ── 2. Classification Metrics Bar Chart ──
fig, ax = plt.subplots(figsize=(9, 5.5))
metrics = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
values = [0.9576, 0.6899, 0.8020, 0.9431]
colors = [ACCENT_B, ACCENT_G, ACCENT_Y, ACCENT_R]

bars = ax.bar(metrics, values, color=colors, width=0.55, edgecolor='#30363d', linewidth=1.2)
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{v*100:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold', color=WHITE)

ax.set_ylim(0, 1.15)
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Classification Performance Metrics\nUnsupervised Isolation Forest — 288,344 Test Records', fontsize=14, fontweight='bold', color=WHITE, pad=15)
ax.axhline(y=0.5, color=SUBTEXT, linestyle='--', linewidth=0.8, alpha=0.5, label='Random baseline')
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('reports/classification_metrics.png', dpi=200, bbox_inches='tight')
plt.close()
print('[2/6] classification_metrics.png')

# ── 3. Latency Metrics ──
fig, ax = plt.subplots(figsize=(9, 5.5))
lat_labels = ['Mean', 'Median', 'p95', 'p99', 'Max']
lat_values = [62.41, 58.74, 82.11, 109.90, 166.76]
lat_colors = [ACCENT_B, ACCENT_G, ACCENT_Y, ACCENT_R, '#da3633']

bars = ax.bar(lat_labels, lat_values, color=lat_colors, width=0.5, edgecolor='#30363d', linewidth=1.2)
for bar, v in zip(bars, lat_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
            f'{v:.1f} ms', ha='center', va='bottom', fontsize=12, fontweight='bold', color=WHITE)

ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
ax.set_title('Streaming Inference Latency (Batch Size = 64)\nPer-Batch Processing Time Distribution', fontsize=14, fontweight='bold', color=WHITE, pad=15)
ax.axhline(y=100, color=ACCENT_R, linestyle='--', linewidth=0.8, alpha=0.5, label='100ms threshold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('reports/latency_metrics.png', dpi=200, bbox_inches='tight')
plt.close()
print('[3/6] latency_metrics.png')

# ── 4. Detection Breakdown (Pie/Donut) ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

# Left: Normal records breakdown
normal_sizes = [232053, 1670]
normal_labels = ['Correctly Identified\n(TN: 232,053)', 'False Alarm\n(FP: 1,670)']
normal_colors = [ACCENT_G, ACCENT_R]
wedges1, texts1, autotexts1 = ax1.pie(normal_sizes, labels=normal_labels, colors=normal_colors,
    autopct='%1.1f%%', startangle=90, pctdistance=0.75,
    textprops={'color': WHITE, 'fontsize': 10},
    wedgeprops={'edgecolor': '#30363d', 'linewidth': 2})
for t in autotexts1:
    t.set_fontweight('bold')
    t.set_fontsize(12)
centre_circle1 = plt.Circle((0,0), 0.50, fc=BG_CARD, edgecolor='#30363d')
ax1.add_artist(centre_circle1)
ax1.set_title('Normal Records (233,723)\nFalse Alarm Rate: 0.71%', fontsize=12, fontweight='bold', color=WHITE, pad=10)

# Right: Attack records breakdown
attack_sizes = [37684, 16937]
attack_labels = ['Detected\n(TP: 37,684)', 'Missed\n(FN: 16,937)']
attack_colors = [ACCENT_B, ACCENT_Y]
wedges2, texts2, autotexts2 = ax2.pie(attack_sizes, labels=attack_labels, colors=attack_colors,
    autopct='%1.1f%%', startangle=90, pctdistance=0.75,
    textprops={'color': WHITE, 'fontsize': 10},
    wedgeprops={'edgecolor': '#30363d', 'linewidth': 2})
for t in autotexts2:
    t.set_fontweight('bold')
    t.set_fontsize(12)
centre_circle2 = plt.Circle((0,0), 0.50, fc=BG_CARD, edgecolor='#30363d')
ax2.add_artist(centre_circle2)
ax2.set_title('Attack Records (54,621)\nRecall: 69.0%', fontsize=12, fontweight='bold', color=WHITE, pad=10)

fig.suptitle('Detection Breakdown — Isolation Forest (Unsupervised)', fontsize=14, fontweight='bold', color=WHITE, y=1.02)
plt.tight_layout()
plt.savefig('reports/detection_breakdown.png', dpi=200, bbox_inches='tight')
plt.close()
print('[4/6] detection_breakdown.png')

# ── 5. Batch vs Streaming Comparison ──
fig, ax = plt.subplots(figsize=(10, 5.5))

categories = ['Throughput\n(rec/s)', 'Latency Mean\n(ms)', 'Memory\nEfficiency', 'Real-Time\nCapability', 'Concept Drift\nAdaptation']
batch_vals = [57000, 4.2, 30, 20, 15]
stream_vals = [3727, 62.4, 95, 95, 85]

x = np.arange(len(categories))
width = 0.32

bars1 = ax.bar(x - width/2, batch_vals, width, label='Batch Processing', color=ACCENT_Y, edgecolor='#30363d', linewidth=1.2, alpha=0.85)
bars2 = ax.bar(x + width/2, stream_vals, width, label='Streaming Processing', color=ACCENT_B, edgecolor='#30363d', linewidth=1.2, alpha=0.85)

for bar, v in zip(bars1, batch_vals):
    label = f'{v:,.0f}' if v > 100 else f'{v:.1f}' if isinstance(v, float) else f'{v}%'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 800, label, ha='center', fontsize=9, fontweight='bold', color=ACCENT_Y)
for bar, v in zip(bars2, stream_vals):
    label = f'{v:,.0f}' if v > 100 else f'{v:.1f}' if isinstance(v, float) else f'{v}%'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 800, label, ha='center', fontsize=9, fontweight='bold', color=ACCENT_B)

ax.set_ylabel('Score / Value', fontsize=12, fontweight='bold')
ax.set_title('Batch vs Streaming Processing Comparison', fontsize=14, fontweight='bold', color=WHITE, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10)
ax.legend(loc='upper right', fontsize=10, facecolor=BG_CARD, edgecolor='#30363d')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('reports/batch_vs_streaming.png', dpi=200, bbox_inches='tight')
plt.close()
print('[5/6] batch_vs_streaming.png')

# ── 6. Summary Dashboard (All KPIs) ──
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('SWaT Anomaly Detection — Performance Summary Dashboard', fontsize=16, fontweight='bold', color=WHITE, y=1.01)

kpis = [
    ('Precision', '95.76%', ACCENT_B, 0.9576),
    ('Recall', '69.00%', ACCENT_G, 0.6899),
    ('F1-Score', '80.20%', ACCENT_Y, 0.8020),
    ('ROC-AUC', '94.31%', ACCENT_R, 0.9431),
    ('False Alarm Rate', '0.71%', ACCENT_G, 0.0071),
    ('Accuracy', '93.55%', ACCENT_B, 0.9355),
]

for ax, (name, display, color, val) in zip(axes.flat, kpis):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor(BG_CARD)
    for spine in ax.spines.values():
        spine.set_color('#30363d')
        spine.set_linewidth(1.5)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(0.5, 0.7, display, ha='center', va='center', fontsize=32, fontweight='bold', color=color, transform=ax.transAxes)
    ax.text(0.5, 0.3, name, ha='center', va='center', fontsize=14, fontweight='bold', color=SUBTEXT, transform=ax.transAxes)

    bar_y = 0.08
    bar_h = 0.06
    ax.add_patch(plt.Rectangle((0.1, bar_y), 0.8, bar_h, transform=ax.transAxes, facecolor='#21262d', edgecolor='none', clip_on=False))
    fill_w = min(val, 1.0) * 0.8
    ax.add_patch(plt.Rectangle((0.1, bar_y), fill_w, bar_h, transform=ax.transAxes, facecolor=color, edgecolor='none', clip_on=False, alpha=0.7))

plt.tight_layout()
plt.savefig('reports/summary_dashboard.png', dpi=200, bbox_inches='tight')
plt.close()
print('[6/6] summary_dashboard.png')

print('\nAll 6 visualizations generated successfully!')
