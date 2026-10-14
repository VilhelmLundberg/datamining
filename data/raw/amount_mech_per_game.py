import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "raw"
PLOT_DIR = PROJECT_ROOT / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

mechanics = pd.read_csv(RAW_DIR / "mechanics.csv")

# Number of mechanics each game has
mechanics_per_game = mechanics.drop("BGGId", axis=1).sum(axis=1)

max_count = mechanics_per_game.max()
plt.figure(figsize=(10, 6))
plt.hist(mechanics_per_game, bins=range(0, max_count + 2), color="steelblue", edgecolor="black", align="left")
plt.title("Distribution of Number of Mechanics per Game")
plt.xlabel("Number of mechanics")
plt.ylabel("Number of games")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(PLOT_DIR / "mechanics_per_game_distribution.jpg", dpi=300)
plt.close()

print(f"Games: {len(mechanics_per_game):,}")
print(f"Median mechanics per game: {mechanics_per_game.median():,.0f}")
print(f"Max mechanics on one game: {max_count:,}")
print(f"Saved: {PLOT_DIR / 'mechanics_per_game_distribution.jpg'}")