import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ['CP1', 'CP4', 'CPN', 'CPS', 'DEH', 'DOH', 'OIE', 'OXH', 'LIH']
new_rmse = [0.31662066721794696, 0.30703188250444746, 0, 0.5235936513493924, 0, 0, 0, 0.38445968458551577, 0]
old_rmse = [0.3298889622854889, 0.33584039352559175, 0.23454275768867155, 0.5504248368597708, 0.5841680355538649, 0.763693819930191, 0.4435802634437575, 0.40493631065796815, 0.6148558504796939]

# Set up the plot
x = np.arange(len(methods))  # Label locations
width = 0.35  # Width of the bars

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, new_rmse, width, label='New Model', color='#1f77b4')
bars2 = ax.bar(x + width/2, old_rmse, width, label='Old Model', color='#ff7f0e')

# Customize the plot
ax.set_xlabel('Buildings')
ax.set_ylabel('cv_rmse')
ax.set_title('Comparison of Epoch Loss values: New Model vs Old Model Across All Buildings')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=45)
ax.legend()

# Add value labels on top of bars, skipping 0 values
for bar in bars1 + bars2:
    height = bar.get_height()
    if height != 0:  # Only annotate non-zero values
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

plt.tight_layout()
plt.show()