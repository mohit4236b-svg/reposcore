"""
Plot calibration reliability diagram from saved metrics.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

with open(os.path.join(DATA_DIR, "metrics_report.json")) as f:
    metrics = json.load(f)

calibration_bins = metrics["holdout_calibration_bins"]
brier = metrics["holdout_brier_score"]

mean_pred = [b["mean_predicted_confidence"] for b in calibration_bins]
frac_pos = [b["observed_fraction_positive"] for b in calibration_bins]

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated', linewidth=1)
ax.plot(mean_pred, frac_pos, 'bo-', label='Observed vs. Predicted', markersize=8)
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('Fraction of Positives (Observed)')
ax.set_title(f'Calibration Curve (Reliability Diagram)\nBrier Score: {brier:.3f}')
ax.legend(loc='upper left')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

for i, (p, o) in enumerate(zip(mean_pred, frac_pos)):
    ax.annotate(f'{p:.0%}', (p, o), textcoords='offset points', xytext=(0, 10), ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, "calibration_curve.png"), dpi=150)
print(f"Saved calibration curve to {os.path.join(DATA_DIR, 'calibration_curve.png')}")