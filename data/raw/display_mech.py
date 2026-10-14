import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

TOP_N = 10

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "raw"
PLOT_DIR = PROJECT_ROOT / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

mechanics = pd.read_csv(RAW_DIR / "mechanics.csv")

# Number of games per mechanic, keep only the top 10
games_per_mechanic = mechanics.drop("BGGId", axis=1).sum()
top_counts = games_per_mechanic.sort_values(ascending=False).head(TOP_N)

plt.figure(figsize=(12, 7))
plt.bar(top_counts.index, top_counts.values, color="steelblue", edgecolor="black")
plt.title(f"Top {TOP_N} Mechanics by Number of Games")
plt.xlabel("Mechanic")
plt.ylabel("Number of games")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(PLOT_DIR / "top10_mechanics_bar.jpg", dpi=300)
plt.close()

print(top_counts.to_string())
print(f"\nSaved: {PLOT_DIR / 'top10_mechanics_bar.jpg'}")