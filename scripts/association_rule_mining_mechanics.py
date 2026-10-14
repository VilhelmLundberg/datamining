import pandas as pd
from pathlib import Path
from mlxtend.frequent_patterns import apriori, association_rules

MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.50
MAX_ITEMSET_LENGTH = 10

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

mechanics = pd.read_csv(RAW_DIR / "mechanics.csv")

print("========================================")
print("MECHANICS ASSOCIATION RULE MINING")
print("========================================")
print(f"Games: {len(mechanics):,}")

mech_cols = [c for c in mechanics.columns if c != "BGGId"]
transactions = mechanics[mech_cols].astype(bool)

print("\n--- RUNNING APRIORI ---")
frequent_itemsets = apriori(
    transactions,
    min_support=MIN_SUPPORT,
    use_colnames=True,
    max_len=MAX_ITEMSET_LENGTH,
    low_memory=True,
)

if frequent_itemsets.empty:
    raise SystemExit("No frequent itemsets found.")

frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)
frequent_itemsets = frequent_itemsets.sort_values("support", ascending=False)

print(f"Frequent itemsets: {len(frequent_itemsets):,}")

print("\n--- GENERATING RULES ---")
rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=MIN_CONFIDENCE,
)

rules = rules[rules["lift"] > 1].sort_values(["lift", "confidence"], ascending=False)

if rules.empty:
    raise SystemExit("No association rules passed the thresholds.")

print(f"Association rules: {len(rules):,}")

rules_output = rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
rules_output["antecedent_names"] = rules_output["antecedents"].apply(lambda itemset: ", ".join(sorted(itemset)))
rules_output["consequent_names"] = rules_output["consequents"].apply(lambda itemset: ", ".join(sorted(itemset)))
rules_output = rules_output[["antecedent_names", "consequent_names", "support", "confidence", "lift"]]

itemsets_output = frequent_itemsets[["itemsets", "support", "length"]].copy()
itemsets_output["mechanic_names"] = itemsets_output["itemsets"].apply(lambda itemset: ", ".join(sorted(itemset)))
itemsets_output = itemsets_output[["mechanic_names", "support", "length"]]

rules_output.to_csv(OUTPUT_DIR / "association_rules_mechanics.csv", index=False)
itemsets_output.to_csv(OUTPUT_DIR / "frequent_itemsets_mechanics.csv", index=False)

print("\n--- TOP RULES ---")
print(rules_output.head(30).to_string(index=False))
print(f"\nSaved: {OUTPUT_DIR / 'association_rules_mechanics.csv'}")
print(f"Saved: {OUTPUT_DIR / 'frequent_itemsets_mechanics.csv'}")