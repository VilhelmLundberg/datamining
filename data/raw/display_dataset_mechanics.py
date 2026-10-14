import pandas as pd
from pathlib import Path

# Load the data
DATA_DIR = Path(__file__).resolve().parent / "outputs"
rules_df = pd.read_csv(DATA_DIR / "association_rules_mechanics.csv")
itemsets_df = pd.read_csv(DATA_DIR / "frequent_itemsets_mechanics.csv")

# Display the results
print("========================================")
print("TOP ASSOCIATION RULES BASED ON MECHANICS")
print("========================================")
print(rules_df.head(30).to_string(index=False))

print("\n========================================")
print("FREQUENT MECHANIC ITEMSETS")
print("========================================")
print(itemsets_df.head(30).to_string(index=False))