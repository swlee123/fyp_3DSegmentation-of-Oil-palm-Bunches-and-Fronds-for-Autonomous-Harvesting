# import pandas as pd
# import matplotlib.pyplot as plt

# # Define data
# data = {
#     "Metrics": ["mAcc", "mIoU", "oAcc"],
#     "Value": [0.6761,0.7163,0.9538],
  
# }

# # Create DataFrame
# df = pd.DataFrame(data)

# # Plot table
# fig, ax = plt.subplots(figsize=(3, 3))
# ax.axis('tight')
# ax.axis('off')

# # Create table
# table = ax.table(
#     cellText=df.values,
#     colLabels=df.columns,
#     cellLoc='center',
#     loc='center'
# )
# table.scale(1, 2)

# # # Optional styling: make summary rows bold
# # for row_index in [-3, -2, -1]:
# #     for col_index in range(len(df.columns)):
# #         cell = table[row_index + 1, col_index]
# #         cell.set_fontsize(10)
# #         cell.set_text_props(weight='bold')

# plt.title("Test Evaluation Metrics", pad=20)
# plt.tight_layout()
# plt.show()


import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Define class-color mapping
legend_elements = [
    Patch(facecolor='blue', edgecolor='black', label='Branch'),
    Patch(facecolor='brown', edgecolor='black', label='Trunk'),
    Patch(facecolor='green', edgecolor='black', label='Ground'),
    Patch(facecolor='red', edgecolor='black', label='Fruit'),
]

# Create figure for legend only
plt.figure(figsize=(4, 2))
plt.legend(handles=legend_elements, loc='center', frameon=True, title='Class Legend')
plt.axis('off')  # Hide axes
plt.tight_layout()
plt.show()
