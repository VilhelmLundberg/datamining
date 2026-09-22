import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_path = "../data/"
plot_path = "../plots/"

#load csv
user_ratings = pd.read_csv(data_path+"user_ratings.csv")
themes = pd.read_csv(data_path+"themes.csv")
subcategories = pd.read_csv(data_path+"subcategories.csv")
ratings_distribution = pd.read_csv(data_path+"ratings_distribution.csv")
publishers = pd.read_csv(data_path+"publishers_reduced.csv")
mechanics = pd.read_csv(data_path+"mechanics.csv")
games = pd.read_csv(data_path+"games.csv")
designers = pd.read_csv(data_path+"designers_reduced.csv")
artists = pd.read_csv(data_path+"artists_reduced.csv")

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
for name,df in dfs.items():
    print(f"{name} shape:", df.shape)
    print(f"\nMissing values in {name}")
    print(df.isnull().sum()[df.isnull().sum() > 0])


#Show the actual user ratings distribution
plt.figure(figsize=(8, 5))
sns.histplot(user_ratings["Rating"], bins=10, binrange=(1,user_ratings["Rating"].max()+1),discrete=True, color='red', edgecolor='black')
plt.title("Distribution of User Ratings")
plt.xlabel("Rating")
plt.ylabel("Frequency")
plt.xticks(np.arange(1, user_ratings["Rating"].max()+1))
plt.savefig(plot_path+"user_rating_distribution.jpg")

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
plt.savefig(plot_path+"top_mechanics.jpg")
