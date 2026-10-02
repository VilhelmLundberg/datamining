from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def handle_missing_values(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    dfs["user_ratings"] = dfs["user_ratings"].dropna(subset=["Username"])

    # Replace missing values with median
    for col in ["ComAgeRec", "LanguageEase"]:
        dfs["games"][col] = dfs["games"][col].fillna(dfs["games"][col].median())

    return dfs


def remove_unneccesary_values(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    dfs["games"] = dfs["games"].drop(columns=["ImagePath", "Description"])
    return dfs


def remove_invalid_users(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    user_stats = dfs["user_ratings"].groupby("Username")["Rating"].agg(rating_count="count", rating_std="std").reset_index()
    valid_users = user_stats[(user_stats["rating_count"] >= 5) & (user_stats["rating_std"] > 0.1)]["Username"]
    dfs["user_ratings"] = dfs["user_ratings"][dfs["user_ratings"]["Username"].isin(valid_users)].copy()
    return dfs


def remove_duplicates(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    dfs["games"] = dfs["games"].drop_duplicates()
    dfs["user_ratings"] = dfs["user_ratings"].drop_duplicates(subset=["Username", "BGGId"], keep="last")
    return dfs


def filter_dataframe(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    # Cap extreme values
    dfs["games"]["MaxPlayers"] = dfs["games"]["MaxPlayers"].clip(upper=13)
    dfs["games"]["MfgPlaytime"] = dfs["games"]["MfgPlaytime"].clip(upper=720)
    dfs["games"]["ComMaxPlaytime"] = dfs["games"]["ComMaxPlaytime"].clip(upper=720)
    dfs["games"]["YearPublished"] = dfs["games"]["YearPublished"].clip(lower=1900, upper=2024)

    # Logarithmic transformations
    for col in ["NumOwned", "NumWant", "NumWish", "NumUserRatings", "NumComments"]:
        dfs["games"][f"{col}_log"] = np.log1p(dfs["games"][col])

    return dfs


def apply_and_report(dfs, function, step_name):
    before = [len(dfs["games"]), len(dfs["user_ratings"]), dfs["user_ratings"]["Username"].nunique()]
    dfs = function(dfs)
    after = [len(dfs["games"]), len(dfs["user_ratings"]), dfs["user_ratings"]["Username"].nunique()]
    removed = [b - a for b, a in zip(before, after)]

    print(f"\n{step_name}")
    print(f"  Games removed:   {removed[0]:,}")
    print(f"  Ratings removed: {removed[1]:,}")
    print(f"  Users removed:   {removed[2]:,}")

    return dfs, [step_name.rstrip("."), *removed]


def save_table(data, columns, title, filename, plot_dir, figsize=(10, 3)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    table = ax.table(cellText=data, colLabels=columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    for col in range(len(columns)):
        table[(0, col)].set_text_props(weight="bold")

    plt.title(title, fontsize=14, weight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(plot_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Paths
    project_root = Path(__file__).resolve().parent.parent
    raw_data_dir = project_root / "data" / "raw"
    processed_data_dir = project_root / "data" / "processed"
    plot_dir = project_root / "plots"
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading raw data...")
    dfs = {
        "games": pd.read_csv(raw_data_dir / "games.csv"),
        "user_ratings": pd.read_csv(raw_data_dir / "user_ratings.csv")
    }

    raw = [len(dfs["games"]), len(dfs["user_ratings"]), dfs["user_ratings"]["Username"].nunique()]
    print(f"Games loaded: {raw[0]:,}")
    print(f"Ratings loaded: {raw[1]:,}")
    print(f"Users loaded: {raw[2]:,}")

    # Preprocessing
    print("\nRunning preprocessing...")
    stats = []

    steps = [
        (handle_missing_values, "Missing values handled."),
        (remove_unneccesary_values, "Unnecessary values removed."),
        (remove_invalid_users, "Invalid users removed."),
        (remove_duplicates, "Duplicates removed."),
        (filter_dataframe, "Filtering and transformations complete.")
    ]

    for function, name in steps:
        dfs, result = apply_and_report(dfs, function, name)
        stats.append(result)

    # Save clean data
    print("\nSaving cleaned data...")
    dfs["games"].to_csv(processed_data_dir / "games_clean.csv", index=False)
    dfs["user_ratings"].to_csv(processed_data_dir / "user_ratings_clean.csv", index=False)

    clean = [len(dfs["games"]), len(dfs["user_ratings"]), dfs["user_ratings"]["Username"].nunique()]
    removed = [r - c for r, c in zip(raw, clean)]

    # Table 1: preprocessing removals
    # Skip unnecessary-column and transformation steps since they don't remove rows
    removal_stats = [row for row in stats if row[0] not in ["Unnecessary values removed", "Filtering and transformations complete"]]
    removal_stats.append(["Total removed", *removed])

    removal_table = [[row[0], *[f"{x:,}" for x in row[1:]]] for row in removal_stats]

    save_table(
        removal_table,
        ["Preprocessing Stage", "Games Removed", "Ratings Removed", "Users Removed"],
        "Data Removed During Preprocessing",
        "preprocessing_removed_table.png",
        plot_dir
    )

    # Table 2: raw vs clean
    labels = ["Games", "Ratings", "Users"]
    summary_table = []

    for label, r, c in zip(labels, raw, clean):
        difference = r - c
        reduction = f"{difference:,} ({difference / r * 100:.2f}%)" if difference else "0"
        summary_table.append([label, f"{r:,}", f"{c:,}", reduction])

    save_table(
        summary_table,
        ["Dataset Measure", "Raw Data", "Clean Data", "Reduction"],
        "Raw vs Clean Dataset",
        "raw_vs_clean_table.png",
        plot_dir,
        figsize=(9, 2.5)
    )

    # Summary
    print("\nPreprocessing complete!")
    print(f"Clean games: {clean[0]:,}")
    print(f"Clean ratings: {clean[1]:,}")
    print(f"Clean users: {clean[2]:,}")
    print(f"\nFiles saved to:\n{processed_data_dir}\n{plot_dir}")