import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

os.makedirs('artifacts', exist_ok=True)
sns.set_theme(style="whitegrid")

# Simulate data
np.random.seed(42)
time_steps = np.arange(0, 1000)

# 1. Feature Drift: Sensor reading with gradual concept drift
# Normal stationary behavior
sensor_base = np.sin(time_steps / 20) + np.random.normal(0, 0.5, 1000)

# Add drift starting at step 400
drift = np.zeros(1000)
drift[400:] = np.linspace(0, 5, 600)  # Gradual upward drift
sensor_data = sensor_base + drift

# 2. Model Performance: False Positive Rate (FPR) increasing due to drift
# Base FPR around 2%
fpr = np.random.normal(0.02, 0.005, 1000)
# FPR increases proportionally to the drift
fpr[400:] += np.linspace(0, 0.4, 600) + np.random.normal(0, 0.02, 600)
fpr = np.clip(fpr, 0, 1) # Keep between 0 and 1

# Retraining point at step 800
retrain_point = 800
# After retraining, FPR drops back to normal
fpr[retrain_point:] = np.random.normal(0.02, 0.005, 200)
fpr = np.clip(fpr, 0, 1)

# Smoothing for better visualization
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

fpr_smooth = smooth(fpr, 20)
# Fix edge effects from smoothing
fpr_smooth[:10] = fpr_smooth[10]
fpr_smooth[-10:] = fpr_smooth[-11]


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

# Top Plot: Sensor Data Drift
ax1.plot(time_steps, sensor_data, color='#388bfd', alpha=0.7, label='Sensor Value')
ax1.axvline(x=400, color='#f85149', linestyle='--', linewidth=2, label='Drift Onset')
ax1.axvline(x=retrain_point, color='#3fb950', linestyle='--', linewidth=2, label='Model Retrained')
ax1.set_title('Sensor Data Stream: Gradual Concept Drift', fontsize=14, fontweight='bold')
ax1.set_ylabel('Sensor Reading')
ax1.legend(loc='upper left')

# Bottom Plot: Model Performance Degradation
ax2.plot(time_steps, fpr_smooth, color='#d29922', linewidth=2.5, label='False Positive Rate (Smoothed)')
ax2.axvline(x=400, color='#f85149', linestyle='--', linewidth=2)
ax2.axvline(x=retrain_point, color='#3fb950', linestyle='--', linewidth=2)
ax2.axvspan(400, retrain_point, color='#f85149', alpha=0.1, label='Degradation Zone')
ax2.axvspan(retrain_point, 1000, color='#3fb950', alpha=0.1, label='Recovery Zone')
ax2.set_title('Model Performance Degradation Over Time', fontsize=14, fontweight='bold')
ax2.set_ylabel('False Positive Rate')
ax2.set_xlabel('Time (Batches)')
ax2.legend(loc='upper left')
ax2.set_ylim(-0.05, 0.5)

plt.tight_layout()
plt.savefig('artifacts/concept_drift.png', dpi=300)
plt.close()

print("Concept drift visual generated successfully.")
