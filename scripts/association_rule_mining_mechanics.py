import pandas as pd
from pathlib import Path

MIN_NUM_RATINGS = 500   # ignore obscure games
TOP_RULES = 5           # one recommendation per top-lift rule

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
merged = games.merge(mechanics, on="BGGId").reset_index(drop=True)

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
picked_games = set()

for _, rule in rules.iterrows():
    if len(rows) >= TOP_RULES:
        break

    antecedent = parse(rule["antecedent_names"])
    consequent = parse(rule["consequent_names"])
    pattern = antecedent + consequent

    if not all(m in mech_set for m in pattern):
        continue

    # Completeness: share of the full pattern a game has, scaled by its rating
    matched = merged[pattern].sum(axis=1)
    completeness = matched / len(pattern)
    score = completeness * merged["BayesAvgRating"]

    candidates = merged.assign(score=score, completeness=completeness)
    candidates = candidates[candidates["completeness"] > 0]
    candidates = candidates.sort_values(["score", "BayesAvgRating"], ascending=False)

    # Skip games already recommended by a higher-lift rule
    best = candidates[~candidates["Name"].isin(picked_games)].head(1)
    if best.empty:
        continue

    g = best.iloc[0]
    picked_games.add(g["Name"])

    rows.append({
        "if_you_like": ", ".join(antecedent),
        "recommended_mechanics": ", ".join(consequent),
        "game": g["Name"],
        "bayes_rating": round(g["BayesAvgRating"], 2),
        "lift": round(rule["lift"], 2),
        "mechanic_match": f"{int(g['completeness'] * len(pattern))}/{len(pattern)}",
        "score": round(g["score"], 2),
    })

recs = pd.DataFrame(rows)
recs.to_csv(OUTPUT_DIR / "game_recommendations.csv", index=False)

print(recs.to_string(index=False))
print(f"\nSaved: {OUTPUT_DIR / 'game_recommendations.csv'}")