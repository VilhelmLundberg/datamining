from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


project_root = Path(__file__).resolve().parent.parent

data_path = project_root / "data" / "raw"
plot_path = project_root / "plots"

#load csv
user_ratings = pd.read_csv(data_path / "user_ratings.csv")
themes = pd.read_csv(data_path / "themes.csv")
subcategories = pd.read_csv(data_path / "subcategories.csv")
ratings_distribution = pd.read_csv(data_path / "ratings_distribution.csv")
publishers = pd.read_csv(data_path / "publishers_reduced.csv")
mechanics = pd.read_csv(data_path / "mechanics.csv")
games = pd.read_csv(data_path / "games.csv")
designers = pd.read_csv(data_path / "designers_reduced.csv")
artists = pd.read_csv(data_path / "artists_reduced.csv")

dfs = {
    "user_ratings": user_ratings,
    "themes": themes,
    "subcategories": subcategories,
    "ratings_distribution": ratings_distribution,
    "publishers": publishers,
    "mechanics": mechanics,
    "games": games,
    "designers": designers,
    "artists": artists
}

#print the shape and missing values for all
#for name,df in dfs.items():
#    print(f"{name} shape:", df.shape)
#    print(f"\nMissing values in {name}")
#    print(df.isnull().sum()[df.isnull().sum() > 0])

# plot missing values (games)

missing = games.isnull().sum()
missing = missing[missing > 0].sort_values()

plt.figure(figsize=(10, 6))
missing.plot(kind="barh")
plt.title("Missing Values in Games Dataset")
plt.xlabel("Number of Missing Values")
plt.ylabel("Variable")
plt.tight_layout()
plt.savefig(plot_path / "games_missing_values.jpg")
plt.close()

#Show the actual user ratings distribution
plt.figure(figsize=(8, 5))
sns.histplot(user_ratings["Rating"], bins=10, binrange=(1,user_ratings["Rating"].max()+1),discrete=True, color='red', edgecolor='black')
plt.title("Distribution of User Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.xticks(np.arange(1, user_ratings["Rating"].max()+1))
plt.savefig(plot_path / "user_rating_distribution.jpg")

#dont display the bggid
only_mechanics = mechanics.drop("BGGId", axis=1)
counts = only_mechanics.sum().sort_values(ascending=True)

#select only for top 20 mechanics
top = 20
top_counts = counts.tail(top)

plt.figure(figsize=(10, 8))
top_counts.plot(kind="bar", color="orange", edgecolor="black")

plt.title(f"Top {top} mechanics", fontsize=14)
plt.xlabel("Frequency", fontsize=12)
plt.ylabel("Mechanic", fontsize=12)
plt.grid(axis="x", linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(plot_path / "top_mechanics.jpg")


# user activity plot

user_activity = user_ratings.groupby("Username").size()

plt.figure(figsize=(8, 5))
sns.histplot(user_activity[user_activity <= 100], bins=50)
plt.axvline(5, linestyle="--", label="Preprocessing threshold (5)")
plt.axvline(10, linestyle="--", label="ARM threshold (10)")
plt.title("Number of Ratings per User")
plt.xlabel("Number of Ratings")
plt.ylabel("Number of Users")
plt.legend()
plt.tight_layout()
plt.savefig(plot_path / "user_activity_distribution.jpg")
plt.close()


# Histograms of other stats with outliers listed

plot_ranges = {
    "MinPlayers": (1, 13),
    "MaxPlayers": (1, 13),
    "MfgPlaytime": (0, 720),
    "ComMaxPlaytime": (0, 720),
    "YearPublished": (1900, 2021)
}

for col, (lower, upper) in plot_ranges.items():
    data = games[col].dropna()
    data_in_range = data[(data >= lower) & (data <= upper)]

    # Calculate values outside displayed range
    outside = ((data < lower) | (data > upper)).sum()
    percentage = outside / len(data) * 100

    plt.figure(figsize=(8, 5))
    sns.histplot(data_in_range, bins=30)

    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

    # Information box
    info = (
        f"Displayed range: {lower}–{upper}\n"
        f"Values outside range: {outside:,}\n"
        f"Outside range: {percentage:.2f}%"
    )

    plt.text(
        0.97, 0.95,
        info,
        transform=plt.gca().transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(plot_path / f"{col}_distribution.jpg", dpi=300)
    plt.close()


# statistics of important variables

important_variables = [
    "YearPublished",
    "MinPlayers",
    "MaxPlayers",
    "MfgPlaytime",
    "ComAgeRec",
    "LanguageEase",
    "NumOwned",
    "NumUserRatings"
]

print("\n--- DESCRIPTIVE STATISTICS ---")
print(games[important_variables].describe().round(2))