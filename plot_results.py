import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file
df = pd.read_csv('path-to-results/results.csv')

# Clean column names by stripping whitespace
df.columns = df.columns.str.strip()

# Set style
plt.style.use('default')
sns.set_theme()

# Create figure and subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('YOLOv11 Training Results', fontsize=16, y=0.95)

# Plot 1: Training Losses
ax1.plot(df['epoch'], df['train/box_loss'], label='box_loss', linewidth=2)
ax1.plot(df['epoch'], df['train/cls_loss'], label='cls_loss', linewidth=2)
ax1.plot(df['epoch'], df['train/dfl_loss'], label='dfl_loss', linewidth=2)
ax1.set_title('Training Losses')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# Plot 2: Validation Losses
ax2.plot(df['epoch'], df['val/box_loss'], label='box_loss', linewidth=2)
ax2.plot(df['epoch'], df['val/cls_loss'], label='cls_loss', linewidth=2)
ax2.plot(df['epoch'], df['val/dfl_loss'], label='dfl_loss', linewidth=2)
ax2.set_title('Validation Losses')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

# Plot 3: Metrics
ax3.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', linewidth=2)
ax3.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', linewidth=2)
ax3.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50', linewidth=2)
ax3.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95', linewidth=2)
ax3.set_title('Metrics')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Value')
ax3.legend()
ax3.grid(True)

# Plot 4: Learning Rate
ax4.plot(df['epoch'], df['lr/pg0'], label='Learning Rate', linewidth=2)
ax4.set_title('Learning Rate')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Learning Rate')
ax4.grid(True)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('path-to-save/training_results.png', dpi=300, bbox_inches='tight')
plt.close() 