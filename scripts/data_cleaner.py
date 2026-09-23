import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def handle_missing_values(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    dfs["user_ratings"] = dfs["user_ratings"].dropna(subset=["Username"])

    #replace missing comagerec and languageease with median value
    #too many missing, to simply remove
    for col in ["ComAgeRec", "LanguageEase"]:
        dfs["games"][col] = dfs["games"][col].fillna(dfs["games"][col].median())

    return dfs

def remove_unneccesary_values(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    dfs["games"] = dfs["games"].drop(columns=["ImagePath"])
    dfs["games"] = dfs["games"].drop(columns=["Description"])
    
    return dfs
    
#remove outliers and create upper/lower bounds
def filter_dataframe(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    
    #rating counts and standard deviation
    user_stats = dfs["user_ratings"].groupby("Username")["Rating"].agg(rating_count="count",rating_std="std").reset_index()

    #Define a valid user
    #more than 5 ratings
    #no zero variance ratings
    valid_users = user_stats[
        (user_stats["rating_count"] >= 5) &
        (user_stats["rating_std"] > 0.1)]["Username"]

    #remove from user ratings based on criteria
    dfs["user_ratings"] = dfs["user_ratings"][user_ratings["Username"].isin(valid_users)].copy()

    #create upper/lower bounds
    dfs["games"]["MaxPlayers"] = dfs["games"]["MaxPlayers"].clip(upper=13)
    dfs["games"]["MfgPlaytime"] = dfs["games"]["MfgPlaytime"].clip(upper=720)
    dfs["games"]["ComMaxPlaytime"] = dfs["games"]["ComMaxPlaytime"].clip(upper=720)
    dfs["games"]["YearPublished"] = dfs["games"]["YearPublished"].clip(lower=1900,upper=2024)

    #logarithmic transformation
    for col in ["NumOwned","NumWant","NumWish","NumUserRatings","NumComments"]:
        dfs["games"][f"{col}_log"] = np.log1p(dfs["games"][col])

    return dfs
