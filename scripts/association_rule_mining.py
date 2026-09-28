import pandas as pd
from pathlib import Path
import time
from mlxtend.frequent_patterns import apriori, association_rules


# ============================================================
# SETTINGS
# ============================================================

MIN_RATING = 7
MIN_USER_RATINGS = 10
MIN_GAME_LIKES = 2000
MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.30
MAX_ITEMSET_LENGTH = 2


# ============================================================
# PATHS + LOAD DATA
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

games = pd.read_csv(DATA_DIR / "games_clean.csv")
ratings = pd.read_csv(DATA_DIR / "user_ratings_clean.csv")

print("========================================")
print("ASSOCIATION RULE MINING")
print("========================================")
print(f"Ratings: {len(ratings):,}")
print(f"Users: {ratings['Username'].nunique():,}")
print(f"Games in ratings: {ratings['BGGId'].nunique():,}")


# ============================================================
# 1. FILTER USERS
# ============================================================

# Only keep users who have rated enough games to form meaningful transactions.
user_rating_counts = ratings.groupby("Username").size()
valid_users = user_rating_counts[user_rating_counts >= MIN_USER_RATINGS].index
ratings_arm = ratings[ratings["Username"].isin(valid_users)].copy()

print("\n--- USER FILTERING ---")
print(f"Minimum ratings required: {MIN_USER_RATINGS}")
print(f"Users remaining: {ratings_arm['Username'].nunique():,}")
print(f"Ratings remaining: {len(ratings_arm):,}")


# ============================================================
# 2. CONVERT RATINGS INTO LIKES
# ============================================================

# ARM uses binary transactions, so Rating >= MIN_RATING is treated as a like.
liked = ratings_arm[ratings_arm["Rating"] >= MIN_RATING].copy()

print("\n--- LIKE FILTERING ---")
print(f"Minimum rating considered a like: {MIN_RATING}")
print(f"Positive ratings remaining: {len(liked):,}")
print(f"Users with at least one liked game: {liked['Username'].nunique():,}")
print(f"Games with at least one like: {liked['BGGId'].nunique():,}")


# ============================================================
# 3. REMOVE RARE GAMES
# ============================================================

# Only keep games liked by at least MIN_GAME_LIKES unique users.
game_like_counts = liked.groupby("BGGId")["Username"].nunique()
valid_games = game_like_counts[game_like_counts >= MIN_GAME_LIKES].index
liked = liked[liked["BGGId"].isin(valid_games)].copy()

print("\n--- GAME FILTERING ---")
print(f"Minimum likes required: {MIN_GAME_LIKES}")
print(f"Games remaining: {liked['BGGId'].nunique():,}")
print(f"Users remaining: {liked['Username'].nunique():,}")
print(f"Positive ratings remaining: {len(liked):,}")


# ============================================================
# 4. CREATE TRANSACTION MATRIX
# ============================================================

basket = liked.assign(Liked=True).pivot_table(
    index="Username", columns="BGGId", values="Liked",
    aggfunc="max", fill_value=False
).astype(bool)

# Convert to sparse representation to reduce memory usage
basket = basket.astype(pd.SparseDtype(bool, False))

# mlxtend requires sparse DataFrames with integer column names
# to use compatible names, so convert BGGIds to strings
basket.columns = basket.columns.astype(str)

number_of_likes = len(liked)
number_of_possible_values = basket.shape[0] * basket.shape[1]
density = number_of_likes / number_of_possible_values

print("\n--- TRANSACTION MATRIX ---")
print(f"Transactions (users): {basket.shape[0]:,}")
print(f"Items (games): {basket.shape[1]:,}")
print(f"Matrix density: {density:.4%}")


# ============================================================
# 5. RUN APRIORI
# ============================================================

print("\n--- RUNNING APRIORI ---")
start_time = time.time()

frequent_itemsets = apriori(
    basket,
    min_support=MIN_SUPPORT,
    use_colnames=True,
    max_len=MAX_ITEMSET_LENGTH,
    low_memory=True
)

elapsed_time = time.time() - start_time
print(f"Apriori runtime: {elapsed_time:.2f} seconds")

frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)
frequent_itemsets = frequent_itemsets.sort_values("support", ascending=False)

print(f"Frequent itemsets found: {len(frequent_itemsets):,}")

if frequent_itemsets.empty:
    print("No frequent itemsets found. Try decreasing MIN_SUPPORT or MIN_GAME_LIKES.")
    exit()


# ============================================================
# 6. GENERATE ASSOCIATION RULES
# ============================================================

print("\n--- GENERATING RULES ---")

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
rules = rules[rules["lift"] > 1].copy()
rules = rules.sort_values(["lift", "confidence"], ascending=False)

print(f"Association rules found: {len(rules):,}")

if rules.empty:
    print("No rules passed the thresholds. Try decreasing MIN_CONFIDENCE or MIN_SUPPORT.")
    exit()


# ============================================================
# 7. CONVERT GAME IDS TO NAMES
# ============================================================

games["BGGId"] = games["BGGId"].astype(str)
game_name_lookup = games.drop_duplicates("BGGId").set_index("BGGId")["Name"].to_dict()

def convert_ids_to_names(itemset):
    return [game_name_lookup.get(game_id, f"Unknown ({game_id})") for game_id in itemset]

rules["antecedent_names"] = rules["antecedents"].apply(convert_ids_to_names)
rules["consequent_names"] = rules["consequents"].apply(convert_ids_to_names)
frequent_itemsets["game_names"] = frequent_itemsets["itemsets"].apply(convert_ids_to_names)


# ============================================================
# 8. SELECT + DISPLAY RESULTS
# ============================================================

rule_results = rules[
    ["antecedent_names", "consequent_names", "support", "confidence", "lift"]
].copy()

print("\n========================================")
print("TOP ASSOCIATION RULES")
print("========================================")
print(rule_results.head(30).to_string(index=False))


# ============================================================
# 9. PREPARE CSV OUTPUT
# ============================================================

rules_output = rule_results.copy()
rules_output["antecedent_names"] = rules_output["antecedent_names"].apply(lambda x: ", ".join(x))
rules_output["consequent_names"] = rules_output["consequent_names"].apply(lambda x: ", ".join(x))

itemsets_output = frequent_itemsets[["game_names", "support", "length"]].copy()
itemsets_output["game_names"] = itemsets_output["game_names"].apply(lambda x: ", ".join(x))


# ============================================================
# 10. SAVE RESULTS
# ============================================================

rules_output.to_csv(OUTPUT_DIR / "association_rules.csv", index=False)
itemsets_output.to_csv(OUTPUT_DIR / "frequent_itemsets.csv", index=False)

print("\n--- FILES CREATED ---")
print(OUTPUT_DIR / "association_rules.csv")
print(OUTPUT_DIR / "frequent_itemsets.csv")