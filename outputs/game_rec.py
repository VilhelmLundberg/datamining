import textwrap
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

TOP_N = 5

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOT_DIR = PROJECT_ROOT / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

recs = pd.read_csv(OUTPUT_DIR / "game_recommendations.csv")

# Remove repeated recommendation pairs, then pick the top five by rating and lift.
recs = recs.drop_duplicates(subset=["recommended_mechanics", "game"], keep="first")
recs = (recs.sort_values(["bayes_rating", "lift"], ascending=False)
            .head(TOP_N)
            .reset_index(drop=True))

ACCENT = "#0ea5e9"
HEADER = "#0f172a"
TEXT = "#334155"
LINE = "#e2e8f0"
BG = "#fafafa"

fig, (left, right) = plt.subplots(
    1, 2, figsize=(16, 1.7 * TOP_N + 2),
    gridspec_kw={"width_ratios": [2, 1], "wspace": 0.03},
    facecolor=BG
)

rows = list(range(TOP_N - 1, -1, -1))
header_y = TOP_N - 0.35

for ax in (left, right):
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, TOP_N)
    ax.axis("off")

# Column x-positions inside the left panel
X_LIKE, X_PROBABLY, X_LIFT = 0.02, 0.36, 0.88

# ---- Left two-thirds: Because you like | You probably like | Lift ----
left.text(X_LIKE, header_y, "Because you like", fontsize=11, fontweight="bold", color=HEADER, va="center")
left.text(X_PROBABLY, header_y, "You probably like", fontsize=11, fontweight="bold", color=HEADER, va="center")
left.text(X_LIFT, header_y, "Lift", fontsize=11, fontweight="bold", color=HEADER, va="center", ha="center")
left.axhline(header_y - 0.4, color=LINE, linewidth=1.2)

for y, r in zip(rows, recs.itertuples()):
    likes = "\n".join(textwrap.wrap(r.if_you_like, width=28))
    probably = "\n".join(textwrap.wrap(r.recommended_mechanics, width=32))

    left.text(X_LIKE, y, likes, fontsize=10, color=TEXT, va="center")
    left.text(X_PROBABLY, y, probably, fontsize=10, color=HEADER, va="center")

    left.scatter(X_LIFT - 0.03, y, s=r.lift * 150, color=ACCENT,
                 edgecolor="white", linewidth=1.5, zorder=3)
    left.text(X_LIFT + 0.02, y, f"{r.lift:.2f}", fontsize=10, color=TEXT, va="center")

    left.axhline(y - 0.5, color=LINE, linewidth=0.8)

# ---- Right third: Recommended game ----
right.text(0.05, header_y, "Recommended game", fontsize=11, fontweight="bold", color=HEADER, va="center")
right.axhline(header_y - 0.4, color=LINE, linewidth=1.2)

for y, game in zip(rows, recs["game"]):
    right.text(0.05, y, "\n".join(textwrap.wrap(game, width=26)),
               fontsize=11, fontweight="bold", color=ACCENT, va="center")
    right.axhline(y - 0.5, color=LINE, linewidth=0.8)

fig.suptitle("Top 5 Game Recommendations", fontsize=16, fontweight="bold",
             color=HEADER, x=0.04, ha="left", y=0.97)
fig.text(0.04, 0.92, "Dot size shows the lift of the mechanic rule behind each pick",
         fontsize=10, color=TEXT, ha="left")

fig.subplots_adjust(left=0.03, right=0.97, top=0.84, bottom=0.04)
plt.savefig(PLOT_DIR / "game_recommendations.jpg", dpi=300, facecolor=BG)
print(f"Saved: {PLOT_DIR / 'game_recommendations.jpg'}")