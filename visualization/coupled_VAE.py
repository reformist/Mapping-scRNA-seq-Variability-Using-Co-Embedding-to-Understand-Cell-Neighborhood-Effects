import matplotlib.pyplot as plt

# Plot the pie chart with maximum text size for visibility
fig, ax = plt.subplots(figsize=(12, 12), facecolor="#f7f7f7")
wedges, texts, autotexts = ax.pie(
    sizes,
    explode=(0.1, 0.1, 0.2),  # Explode slices for emphasis
    labels=labels,
    colors=colors,
    autopct="%1.1f%%",
    startangle=140,
    textprops={"fontsize": 24, "weight": "bold"},  # Maximum text size for labels
    wedgeprops={"edgecolor": "black", "linewidth": 2},
    pctdistance=0.7,
)

# Style for percentage labels
for autotext in autotexts:
    autotext.set_color("black")
    autotext.set_fontsize(28)  # Maximum size for percentage text
    autotext.set_weight("bold")

# Add title
ax.set_title(
    "Market Breakdown: Stem Cell Therapy (2033)",
    fontsize=30,
    fontweight="bold",
    color="#2f4b7c",
    pad=50,
)

# Remove axes
ax.axis("equal")  # Ensure the pie is drawn as a circle

# Show the chart
plt.tight_layout()
plt.show()
