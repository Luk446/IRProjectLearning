import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("controllers/supervisorGA_starter/data/robot_position_20251109-001346.csv")

# Fitness evolution
df.groupby("generation")["fitness"].mean().plot(title="Average Fitness per Generation")

# Positions
plt.scatter(df["x"], df["y"], c=df["fitness"], cmap="viridis")
plt.colorbar(label="Fitness")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
