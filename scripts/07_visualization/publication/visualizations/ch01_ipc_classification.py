"""
Figure 1: IPC Classification System
Publication-grade visualization of the 5-phase IPC scale

NO HARDCODED DATA - Uses verified crisis rate from DATA_SCHEMA
Date: 2026-01-04
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
import json
from config import BASE_DIR

# Output directory - save to dissertation folder for self-containment
BASE_DIR = Path(rstr(BASE_DIR))
DATA_DIR = BASE_DIR / "Dissertation Write Up" / "DATA_INVENTORIES"
OUTPUT_DIR = BASE_DIR / "Dissertation Write Up" / "LATEX_DISSERTATION" / "figures" / "ch01_introduction"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load REAL crisis rate from JSON
with open(DATA_DIR / "MASTER_METRICS_ALL_MODELS.json", 'r') as f:
    data = json.load(f)
CRISIS_RATE = data['cascade']['production']['crisis_rate']

print(f"Loaded REAL crisis rate from JSON: {CRISIS_RATE:.6f} ({CRISIS_RATE*100:.2f}%)")

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

# IPC Phases (OFFICIAL IPC CLASSIFICATION - NOT HARDCODED DATA)
PHASES = [
    {"phase": 1, "name": "Minimal", "color": "#4CAF50", "description": "Food secure"},
    {"phase": 2, "name": "Stressed", "color": "#FFC107", "description": "Borderline food insecure"},
    {"phase": 3, "name": "Crisis", "color": "#FF9800", "description": "Acute food & livelihood crisis"},
    {"phase": 4, "name": "Emergency", "color": "#F44336", "description": "Humanitarian emergency"},
    {"phase": 5, "name": "Famine", "color": "#B71C1C", "description": "Catastrophe/famine"}
]

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))

# Draw IPC phases as horizontal bars
y_position = 0
bar_height = 0.8
spacing = 0.3

for i, phase_info in enumerate(PHASES):
    # Draw rectangle
    rect = patches.Rectangle(
        (0, y_position),
        phase_info['phase'],
        bar_height,
        linewidth=2,
        edgecolor='black',
        facecolor=phase_info['color'],
        alpha=0.8
    )
    ax.add_patch(rect)

    # Add phase number
    ax.text(
        phase_info['phase'] / 2,
        y_position + bar_height / 2,
        f"Phase {phase_info['phase']}",
        ha='center',
        va='center',
        fontsize=14,
        fontweight='bold',
        color='white' if phase_info['phase'] >= 3 else 'black'
    )

    # Add phase name and description (right side)
    ax.text(
        5.5,
        y_position + bar_height / 2,
        f"{phase_info['name']}: {phase_info['description']}",
        ha='left',
        va='center',
        fontsize=11,
        fontweight='bold' if phase_info['phase'] >= 3 else 'normal'
    )

    y_position -= (bar_height + spacing)

# Add crisis threshold line
crisis_y = PHASES[1]["phase"] * -1 * (bar_height + spacing) + bar_height
ax.plot([0, 11], [crisis_y, crisis_y], 'k--', linewidth=2, label='Crisis Threshold (IPC ≥ 3)')

# Add crisis threshold annotation
ax.annotate(
    'CRISIS THRESHOLD\n(Binary classification target)',
    xy=(5.5, crisis_y),
    xytext=(8, crisis_y + 0.5),
    fontsize=11,
    fontweight='bold',
    color='red',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
    arrowprops=dict(arrowstyle='->', color='red', lw=2)
)

# Add crisis rate statistic
ax.text(
    8, -4.5,
    f'Crisis rate in dataset:\n{CRISIS_RATE*100:.1f}% of observations\n(IPC Phase 3+)',
    fontsize=10,
    bbox=dict(boxstyle='round,pad=0.7', facecolor='lightgray', alpha=0.5),
    ha='left'
)

# Title
ax.text(
    5.5, 1.5,
    'IPC Food Security Phase Classification System',
    ha='center',
    fontsize=16,
    fontweight='bold'
)

# Subtitle
ax.text(
    5.5, 1.1,
    'Standardized scale for acute food insecurity classification (Phases 1-5)',
    ha='center',
    fontsize=11,
    style='italic'
)

# Configure axes
ax.set_xlim(0, 11)
ax.set_ylim(-5, 2)
ax.axis('off')

# Add source note
ax.text(
    0.5, -4.8,
    'Source: IPC Global Partners (https://www.ipcinfo.org). Phase definitions from IPC Technical Manual v3.1',
    fontsize=8,
    style='italic',
    color='gray'
)

plt.tight_layout()

# Save figure
output_file = OUTPUT_DIR / "ch01_ipc_classification.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"[OK] Saved: {output_file}")

# Also save PNG for quick preview
output_file_png = OUTPUT_DIR / "ch01_ipc_classification.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', format='png')
print(f"[OK] Saved: {output_file_png}")

plt.close()

print("\n" + "="*80)
print("FIGURE 1 COMPLETE: IPC Classification System")
print("="*80)
