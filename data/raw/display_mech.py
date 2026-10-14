import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

TOP_N = 10

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOT_DIR = PROJECT_ROOT / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

rules = pd.read_csv(OUTPUT_DIR / "association_rules_mechanics.csv")

top = rules.sort_values("lift", ascending=False).head(TOP_N).reset_index(drop=True)
top["id"] = top.index + 1
top["rule"] = top["antecedent_names"] + " -> " + top["consequent_names"]

fig, (ax, tax) = plt.subplots(
	1,
	2,
	figsize=(16, 7),
	gridspec_kw={"width_ratios": [1.3, 1]},
)

lift_min = top["lift"].min()
lift_span = max(top["lift"].max() - lift_min, 1e-9)
sizes = 140 + 860 * ((top["lift"] - lift_min) / lift_span) ** 1.5

sc = ax.scatter(
	top["support"],
	top["confidence"],
	s=sizes,
	c=top["lift"],
	cmap="viridis",
	alpha=0.7,
	edgecolor="black",
	linewidths=0.8,
)

# Number each dot so the labels stay clear even when dots are close
for r in top.itertuples():
	ax.annotate(
		str(r.id),
		(r.support, r.confidence),
		ha="center",
		va="center",
		fontsize=9,
		fontweight="bold",
	)

fig.colorbar(sc, ax=ax, label="Lift")
ax.set_xlabel("Support")
ax.set_ylabel("Confidence")
ax.set_title(f"Top {TOP_N} Mechanics Association Rules (dot size = lift)")
ax.grid(alpha=0.3)

# Rule legend on the right
tax.axis("off")
legend_lines = [
	f"{r.id}. {r.rule}\n    lift {r.lift:.2f} | support {r.support:.3f} | confidence {r.confidence:.2f}"
	for r in top.itertuples()
]
tax.text(
	0,
	1,
	"Top mechanics rules\n\n" + "\n\n".join(legend_lines),
	va="top",
	fontsize=9,
	family="monospace",
)

plt.tight_layout()
output_file = PLOT_DIR / "top10_mechanics_rules_grid.jpg"
plt.savefig(output_file, dpi=300, bbox_inches="tight")
plt.close()

print(top[["antecedent_names", "consequent_names", "support", "confidence", "lift"]].to_string(index=False))
print(f"\nSaved: {output_file}")