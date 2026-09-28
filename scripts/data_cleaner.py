from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def handle_missing_values(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    dfs["user_ratings"] = dfs["user_ratings"].dropna(subset=["Username"])

    #replace missing comagerec and languageease with median value
    #too many missing, to simply remove
    for col in ["ComAgeRec", "LanguageEase"]:
        dfs["games"][col] = dfs["games"][col].fillna(dfs["games"][col].median())

    return dfs

def remove_unneccesary_values(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    dfs["games"] = dfs["games"].drop(columns=["ImagePath"])
    dfs["games"] = dfs["games"].drop(columns=["Description"])
    
    return dfs
    
#remove outliers and create upper/lower bounds
def filter_dataframe(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    
    #rating counts and standard deviation
    user_stats = dfs["user_ratings"].groupby("Username")["Rating"].agg(rating_count="count",rating_std="std").reset_index()

    #Define a valid user
    #more than 5 ratings
    #no zero variance ratings
    valid_users = user_stats[
        (user_stats["rating_count"] >= 5) &
        (user_stats["rating_std"] > 0.1)]["Username"]

    #remove from user ratings based on criteria
    dfs["user_ratings"] = dfs["user_ratings"][dfs["user_ratings"]["Username"].isin(valid_users)].copy()

    #create upper/lower bounds
    dfs["games"]["MaxPlayers"] = dfs["games"]["MaxPlayers"].clip(upper=13)
    dfs["games"]["MfgPlaytime"] = dfs["games"]["MfgPlaytime"].clip(upper=720)
    dfs["games"]["ComMaxPlaytime"] = dfs["games"]["ComMaxPlaytime"].clip(upper=720)
    dfs["games"]["YearPublished"] = dfs["games"]["YearPublished"].clip(lower=1900,upper=2024)

    #logarithmic transformation
    for col in ["NumOwned","NumWant","NumWish","NumUserRatings","NumComments"]:
        dfs["games"][f"{col}_log"] = np.log1p(dfs["games"][col])

    return dfs

def remove_duplicates(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    # Remove completely duplicated rows
    dfs["games"] = dfs["games"].drop_duplicates()

    # A user should only have one rating for a particular game.
    # If duplicates exist, keep the last occurrence.
    dfs["user_ratings"] = dfs["user_ratings"].drop_duplicates(
        subset=["Username", "BGGId"],
        keep="last"
    )

    return dfs


def apply_and_report(dfs, function, step_name):
    # Before
    games_before = len(dfs["games"])
    ratings_before = len(dfs["user_ratings"])
    users_before = dfs["user_ratings"]["Username"].nunique()
    game_columns_before = len(dfs["games"].columns)
    rating_columns_before = len(dfs["user_ratings"].columns)

    # Apply function
    dfs = function(dfs)

    # After
    games_after = len(dfs["games"])
    ratings_after = len(dfs["user_ratings"])
    users_after = dfs["user_ratings"]["Username"].nunique()
    game_columns_after = len(dfs["games"].columns)
    rating_columns_after = len(dfs["user_ratings"].columns)

    # Report
    print(f"\n{step_name}")
    print(f"  Games removed:         {games_before - games_after:,}")
    print(f"  Ratings removed:       {ratings_before - ratings_after:,}")
    print(f"  Users removed:         {users_before - users_after:,}")
    print(f"  Game columns changed:  {game_columns_after - game_columns_before:+,}")
    print(f"  Rating columns changed:{rating_columns_after - rating_columns_before:+,}")

    return dfs

if __name__ == "__main__":

    # =========================================================
    # SET UP PATHS
    # =========================================================

    # .parent = scripts/
    # .parent.parent = project root
    project_root = Path(__file__).resolve().parent.parent

    raw_data_dir = project_root / "data" / "raw"
    processed_data_dir = project_root / "data" / "processed"

    # Create processed folder if it doesn't already exist
    processed_data_dir.mkdir(parents=True, exist_ok=True)


    # =========================================================
    # LOAD RAW DATA
    # =========================================================

    print("Loading raw data...")

    dfs = {
        "games": pd.read_csv(raw_data_dir / "games.csv"),
        "user_ratings": pd.read_csv(raw_data_dir / "user_ratings.csv")
    }

    print(f"Games loaded: {len(dfs['games']):,}")
    print(f"Ratings loaded: {len(dfs['user_ratings']):,}")
    print(f"Users loaded: {dfs['user_ratings']['Username'].nunique():,}")


    # =========================================================
    # RUN PREPROCESSING
    # =========================================================

    print("\nRunning preprocessing...")

    dfs = apply_and_report(dfs, handle_missing_values, "Missing values handled.")

    dfs = apply_and_report(dfs, remove_unneccesary_values, "Unnecessary values removed.")

    dfs = apply_and_report(dfs, remove_duplicates, "Duplicates removed.")

    dfs = apply_and_report(dfs, filter_dataframe, "Filtering and transformations complete.")


    # =========================================================
    # SAVE CLEAN DATA
    # =========================================================

    print("\nSaving cleaned data...")

    dfs["games"].to_csv(
        processed_data_dir / "games_clean.csv",
        index=False
    )

    dfs["user_ratings"].to_csv(
        processed_data_dir / "user_ratings_clean.csv",
        index=False
    )


    # =========================================================
    # SUMMARY
    # =========================================================

    print("\nPreprocessing complete!")

    print(f"Clean games: {len(dfs['games']):,}")
    print(f"Clean ratings: {len(dfs['user_ratings']):,}")
    print(
        f"Clean users: "
        f"{dfs['user_ratings']['Username'].nunique():,}"
    )

    print("\nFiles saved to:")
    print(processed_data_dir / "games_clean.csv")
    print(processed_data_dir / "user_ratings_clean.csv")