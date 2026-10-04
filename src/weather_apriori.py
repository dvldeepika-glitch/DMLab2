import os
import pandas as pd
from scipy.io import arff
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


# -----------------------------
# Load weather.nominal dataset
# -----------------------------
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(base_dir, "datasets", "weather.nominal.arff")

data, meta = arff.loadarff(file_path)
df = pd.DataFrame(data)

# Convert byte values to strings
for column in df.columns:
    df[column] = df[column].apply(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
    )

print("=" * 70)
print("DATA MINING LAB 2 - APRIORI ASSOCIATION RULE MINING")
print("=" * 70)

print("\nDataset: weather.nominal.arff")
print("Number of instances:", len(df))
print("Number of attributes:", len(df.columns))

print("\nDataset Preview:")
print(df)

# Convert each row into a transaction
transactions = []

for _, row in df.iterrows():
    transaction = []

    for column in df.columns:
        transaction.append(f"{column}={row[column]}")

    transactions.append(transaction)

# One-hot encode transactions
encoder = TransactionEncoder()
encoded_data = encoder.fit(transactions).transform(transactions)

transaction_df = pd.DataFrame(
    encoded_data,
    columns=encoder.columns_
)

# --------------------------------
# Apriori parameters
# --------------------------------
MIN_SUPPORT = 0.20
MIN_CONFIDENCE = 0.80
NUM_RULES = 10

print("\nApriori Parameters:")
print("Minimum Support   :", MIN_SUPPORT)
print("Minimum Confidence:", MIN_CONFIDENCE)
print("Number of Rules   :", NUM_RULES)

# Generate frequent itemsets
frequent_itemsets = apriori(
    transaction_df,
    min_support=MIN_SUPPORT,
    use_colnames=True
)

print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Generate association rules
if len(frequent_itemsets) > 0:
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE
    )

    if len(rules) > 0:
        rules = rules.sort_values(
            by=["confidence", "lift"],
            ascending=False
        )

        rules = rules.head(NUM_RULES)

        print("\n" + "=" * 70)
        print("TOP ASSOCIATION RULES")
        print("=" * 70)

        for index, rule in rules.iterrows():

            antecedent = ", ".join(sorted(list(rule["antecedents"])))
            consequent = ", ".join(sorted(list(rule["consequents"])))

            print(
                f"\nRule {index + 1}: "
                f"{antecedent} ==> {consequent}"
            )

            print(
                f"Support    : {rule['support']:.4f}"
            )

            print(
                f"Confidence : {rule['confidence']:.4f}"
            )

            print(
                f"Lift       : {rule['lift']:.4f}"
            )

    else:
        print("\nNo association rules found.")
else:
    print("\nNo frequent itemsets found.")

print("\n" + "=" * 70)
print("WEATHER APRIORI EXPERIMENT COMPLETED")
print("=" * 70)
