import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOT_DIR = PROJECT_ROOT / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

rules = pd.read_csv(OUTPUT_DIR / "association_rules_mechanics.csv")

top = rules.sort_values("lift", ascending=False).head(5).reset_index(drop=True)
top["id"] = top.index + 1
top["rule"] = top["antecedent_names"] + " -> " + top["consequent_names"]

fig, (ax, tax) = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.3, 1]})

sc = ax.scatter(
    top["support"], top["confidence"],
    s=top["lift"] * 80, c=top["lift"],
    cmap="viridis", alpha=0.7, edgecolor="black", linewidths=0.8
)

# Number each dot so the labels stay clear even when dots are close
for r in top.itertuples():
    ax.annotate(str(r.id), (r.support, r.confidence),
                ha="center", va="center", fontsize=9, fontweight="bold")

fig.colorbar(sc, ax=ax, label="Lift")
ax.set_xlabel("Support")
ax.set_ylabel("Confidence")
ax.set_title("Top 10 Mechanic Association Rules (dot size = lift)")
ax.grid(alpha=0.3)

# Rule legend on the right
tax.axis("off")
legend = "\n\n".join(
    f"{r.id}. {r.rule}\n    lift {r.lift:.2f} | support {r.support:.3f} | confidence {r.confidence:.2f}"
    for r in top.itertuples()
)
tax.text(0, 1, legend, va="top", fontsize=9, family="monospace")

plt.tight_layout()
plt.savefig(PLOT_DIR / "top10_rules_grid.jpg", dpi=300)
print(f"Saved: {PLOT_DIR / 'top10_rules_grid.jpg'}")