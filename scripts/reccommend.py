import pandas as pd
from pathlib import Path

MIN_NUM_RATINGS = 500   # ignore obscure games with few ratings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

rules = pd.read_csv(OUTPUT_DIR / "association_rules_mechanics.csv")
rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
mechanics = pd.read_csv(RAW_DIR / "mechanics.csv")
games = pd.read_csv(
    RAW_DIR / "games.csv",
    usecols=["BGGId", "Name", "BayesAvgRating", "NumUserRatings"]
)

mech_cols = [c for c in mechanics.columns if c != "BGGId"]
mechanics[mech_cols] = mechanics[mech_cols].astype(bool)
mech_set = set(mech_cols)

games = games[games["NumUserRatings"] >= MIN_NUM_RATINGS]
merged = games.merge(mechanics, on="BGGId")

def parse(names):
    # Rebuild mechanic names from the comma-joined string,
    # since some names contain commas themselves
    tokens = names.split(", ")
    result, i = [], 0
    while i < len(tokens):
        for j in range(len(tokens), i, -1):
            candidate = ", ".join(tokens[i:j])
            if candidate in mech_set:
                result.append(candidate)
                i = j
                break
        else:
            result.append(tokens[i])
            i += 1
    return result

rows = []
seen_recommendations = set()

for _, rule in rules.iterrows():
    antecedent = parse(rule["antecedent_names"])
    consequent = parse(rule["consequent_names"])

    if not all(m in mech_set for m in consequent):
        continue

    recommendation_key = ", ".join(consequent)

    if recommendation_key in seen_recommendations:
        continue

    # Keep only the best game for each unique recommendation.
    matches = merged[merged[consequent].all(axis=1)]
    best = matches.sort_values("BayesAvgRating", ascending=False).head(1)

    if best.empty:
        continue

    g = best.iloc[0]
    rows.append({
        "if_you_like": ", ".join(antecedent),
        "recommended_mechanics": recommendation_key,
        "game": g["Name"],
        "bayes_rating": round(g["BayesAvgRating"], 2),
        "lift": round(rule["lift"], 2),
    })
    seen_recommendations.add(recommendation_key)

recs = pd.DataFrame(rows)
recs = recs.drop_duplicates(keep="first")
recs.to_csv(OUTPUT_DIR / "game_recommendations.csv", index=False)

print(recs.to_string(index=False))
print(f"\nSaved: {OUTPUT_DIR / 'game_recommendations.csv'}")