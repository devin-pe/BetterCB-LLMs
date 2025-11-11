import matplotlib.pyplot as plt
import numpy as np

# Data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])  # Number of epochs
y = np.array([2958.0796, 1040.4130, 759.3093, 501.7220, 449.7861, 438.9561, 469.9565, 505.7590])  # Model perplexity

# Create figure
plt.figure(figsize=(6, 4))
plt.plot(
    x, y,
    marker='o',
    linestyle='-',
    color='#2ca02c',
    linewidth=2,
    markersize=5,
    #label='Model Perplexity'
)

# Labels and title
plt.xlabel('Number of Epochs', fontsize=10)
plt.ylabel('Perplexity', fontsize=10)
plt.title('Perplexity over Training Epochs', fontsize=11, pad=8)

# Customize ticks and grid
plt.xticks(x, fontsize=8)
plt.yticks(fontsize=8)
plt.grid(True, which='major', linestyle='--', alpha=0.6)

# Annotate lowest perplexity
min_idx = np.argmin(y)
plt.scatter(x[min_idx], y[min_idx], color='red', s=50, zorder=3)
plt.text(
    x[min_idx], y[min_idx] + 20,
    f"Lowest: {y[min_idx]:.1f}",
    color='red', fontsize=8, ha='center'
)

# Legend and layout
#plt.legend(frameon=False, fontsize=8)
plt.tight_layout()

plt.savefig("perplexity_vs_epochs.svg", bbox_inches='tight')

plt.show()
