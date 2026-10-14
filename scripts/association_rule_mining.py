    import time
    from pathlib import Path
    import pandas as pd
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
    # LOAD DATA
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
    print(f"Games: {ratings['BGGId'].nunique():,}")


    # ============================================================
    # 1. FILTER USERS
    # ============================================================

    # Keep users with enough ratings to form meaningful transactions
    user_counts = ratings.groupby("Username").size()
    valid_users = user_counts[user_counts >= MIN_USER_RATINGS].index
    ratings_arm = ratings[ratings["Username"].isin(valid_users)]

    print("\n--- USER FILTERING ---")
    print(f"Minimum ratings: {MIN_USER_RATINGS}")
    print(f"Users remaining: {ratings_arm['Username'].nunique():,}")
    print(f"Ratings remaining: {len(ratings_arm):,}")


    # ============================================================
    # 2. CONVERT RATINGS TO LIKES
    # ============================================================

    # A rating >= MIN_RATING is considered a positive preference
    liked = ratings_arm[ratings_arm["Rating"] >= MIN_RATING]

    print("\n--- LIKE FILTERING ---")
    print(f"Minimum rating considered a like: {MIN_RATING}")
    print(f"Positive ratings: {len(liked):,}")
    print(f"Users: {liked['Username'].nunique():,}")
    print(f"Games: {liked['BGGId'].nunique():,}")


    # ============================================================
    # 3. REMOVE RARE GAMES
    # ============================================================

    # Keep games liked by at least MIN_GAME_LIKES unique users
    game_counts = liked.groupby("BGGId")["Username"].nunique()
    valid_games = game_counts[game_counts >= MIN_GAME_LIKES].index
    liked = liked[liked["BGGId"].isin(valid_games)]

    print("\n--- GAME FILTERING ---")
    print(f"Minimum likes: {MIN_GAME_LIKES}")
    print(f"Games remaining: {liked['BGGId'].nunique():,}")
    print(f"Users remaining: {liked['Username'].nunique():,}")
    print(f"Positive ratings: {len(liked):,}")


    # ============================================================
    # 4. CREATE TRANSACTION MATRIX
    # ============================================================

    # Rows = users, columns = games, True = user liked the game
    basket = liked.assign(Liked=True).pivot_table(
        index="Username", columns="BGGId", values="Liked",
        aggfunc="max", fill_value=False
    ).astype(pd.SparseDtype(bool, False))

    # mlxtend requires compatible column names for sparse DataFrames
    basket.columns = basket.columns.astype(str)

    density = len(liked) / (basket.shape[0] * basket.shape[1])

    print("\n--- TRANSACTION MATRIX ---")
    print(f"Transactions: {basket.shape[0]:,}")
    print(f"Items: {basket.shape[1]:,}")
    print(f"Density: {density:.4%}")


    # ============================================================
    # 5. RUN APRIORI
    # ============================================================

    print("\n--- RUNNING APRIORI ---")
    start = time.perf_counter()

    frequent_itemsets = apriori(
        basket,
        min_support=MIN_SUPPORT,
        use_colnames=True,
        max_len=MAX_ITEMSET_LENGTH,
        low_memory=True
    )

    runtime = time.perf_counter() - start
    print(f"Runtime: {runtime:.2f} seconds")

    if frequent_itemsets.empty:
        raise SystemExit("No frequent itemsets found.")

    frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)
    frequent_itemsets = frequent_itemsets.sort_values("support", ascending=False)

    print(f"Frequent itemsets: {len(frequent_itemsets):,}")


    # ============================================================
    # 6. GENERATE ASSOCIATION RULES
    # ============================================================

    print("\n--- GENERATING RULES ---")

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE
    )

    # Keep only positive associations and sort strongest first
    rules = rules[rules["lift"] > 1].sort_values(
        ["lift", "confidence"], ascending=False
    )

    if rules.empty:
        raise SystemExit("No association rules passed the thresholds.")

    print(f"Association rules: {len(rules):,}")


    # ============================================================
    # 7. CONVERT GAME IDS TO NAMES
    # ============================================================

    games["BGGId"] = games["BGGId"].astype(str)
    game_names = games.drop_duplicates("BGGId").set_index("BGGId")["Name"].to_dict()

    def names(itemset):
        return [game_names.get(game_id, f"Unknown ({game_id})") for game_id in itemset]

    rules["antecedent_names"] = rules["antecedents"].apply(names)
    rules["consequent_names"] = rules["consequents"].apply(names)
    frequent_itemsets["game_names"] = frequent_itemsets["itemsets"].apply(names)


    # ============================================================
    # 8. DISPLAY RESULTS
    # ============================================================

    rule_results = rules[
        ["antecedent_names", "consequent_names", "support", "confidence", "lift"]
    ].copy()

    print("\n========================================")
    print("TOP ASSOCIATION RULES")
    print("========================================")
    print(rule_results.head(30).to_string(index=False))


    # ============================================================
    # 9. PREPARE OUTPUT
    # ============================================================

    rules_output = rule_results.copy()

    for col in ["antecedent_names", "consequent_names"]:
        rules_output[col] = rules_output[col].apply(", ".join)

    itemsets_output = frequent_itemsets[
        ["game_names", "support", "length"]
    ].copy()

    itemsets_output["game_names"] = itemsets_output["game_names"].apply(", ".join)


    # ============================================================
    # 10. SAVE RESULTS
    # ============================================================

    rules_output.to_csv(OUTPUT_DIR / "association_rules.csv", index=False)
    itemsets_output.to_csv(OUTPUT_DIR / "frequent_itemsets.csv", index=False)

    print("\n--- FILES CREATED ---")
    print(OUTPUT_DIR / "association_rules.csv")
    print(OUTPUT_DIR / "frequent_itemsets.csv")