import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_association_rules_csv(csv_path, plot_dir):
    rules = pd.read_csv(csv_path)

    required = {"antecedent_names", "consequent_names", "support", "confidence", "lift"}
    missing = required - set(rules.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    top = rules.sort_values("lift", ascending=False).head(10).reset_index(drop=True)
    top["id"] = top.index + 1
    top["rule"] = top["antecedent_names"] + " -> " + top["consequent_names"]

    fig, (ax, right) = plt.subplots(
        1,
        2,
        figsize=(16, 7),
        gridspec_kw={"width_ratios": [1.3, 1]},
    )

    lift_min = top["lift"].min()
    lift_max = top["lift"].max()
    lift_span = lift_max - lift_min if lift_max > lift_min else 1.0
    sizes = 150 + 1200 * ((top["lift"] - lift_min) / lift_span)

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

    for row in top.itertuples():
        ax.annotate(
            str(row.id),
            (row.support, row.confidence),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    fig.colorbar(sc, ax=ax, label="Lift")
    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    ax.set_title("Top 10 Association Rules: Support vs Confidence")
    ax.grid(alpha=0.3)

    right.axis("off")
    legend_lines = [
        f"{row.id}. {row.rule}\n    lift {row.lift:.2f} | support {row.support:.3f} | confidence {row.confidence:.2f}"
        for row in top.itertuples()
    ]
    right.text(0, 1, "Top rules\n\n" + "\n\n".join(legend_lines), va="top", fontsize=9, family="monospace")

    plot_dir.mkdir(parents=True, exist_ok=True)
    output_path = plot_dir / "association_rules_visualization.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")
    return output_path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    plot_association_rules_csv(
        project_root / "outputs" / "association_rules.csv",
        project_root / "plots",
    )
