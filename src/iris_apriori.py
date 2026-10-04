import os
import pandas as pd
from scipy.io import arff
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


# ---------------------------------------
# Load iris.arff
# ---------------------------------------
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(base_dir, "datasets", "iris.arff")

data, meta = arff.loadarff(file_path)
df = pd.DataFrame(data)

# Convert byte values to strings
for column in df.columns:
    df[column] = df[column].apply(
        lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
    )

print("=" * 75)
print("DATA MINING LAB 2 - IRIS APRIORI")
print("=" * 75)

print("\nDataset: iris.arff")
print("Instances:", len(df))
print("Attributes:", len(df.columns))

print("\nOriginal Dataset:")
print(df.head())

# ----------------------------------------------------
# PART 1 - Before Discretization
# ----------------------------------------------------

print("\n" + "=" * 75)
print("PART 1: APRIORI BEFORE DISCRETIZATION")
print("=" * 75)

print(
    "\nThe Iris dataset contains continuous numeric attributes."
)
print(
    "Numeric values are mostly unique, so direct association "
    "mining is sparse and difficult to interpret."
)

# Convert numeric values to exact-value transactions
transactions_before = []

for _, row in df.iterrows():

    transaction = []

    for column in df.columns:
        transaction.append(f"{column}={row[column]}")

    transactions_before.append(transaction)

encoder_before = TransactionEncoder()

encoded_before = encoder_before.fit(
    transactions_before
).transform(transactions_before)

before_df = pd.DataFrame(
    encoded_before,
    columns=encoder_before.columns_
)

frequent_before = apriori(
    before_df,
    min_support=0.20,
    use_colnames=True
)

print("\nMinimum Support: 0.20")

if len(frequent_before) > 0:

    rules_before = association_rules(
        frequent_before,
        metric="confidence",
        min_threshold=0.80
    )

    if len(rules_before) > 0:

        print("\nRules Before Discretization:")

        rules_before = rules_before.sort_values(
            by="confidence",
            ascending=False
        ).head(10)

        for _, rule in rules_before.iterrows():

            antecedent = ", ".join(
                sorted(list(rule["antecedents"]))
            )

            consequent = ", ".join(
                sorted(list(rule["consequents"]))
            )

            print(
                f"{antecedent} ==> {consequent}"
            )

            print(
                f"Support={rule['support']:.4f}, "
                f"Confidence={rule['confidence']:.4f}, "
                f"Lift={rule['lift']:.4f}"
            )

    else:
        print("\nNo strong rules found before discretization.")

else:
    print("\nNo frequent itemsets found.")

# ----------------------------------------------------
# PART 2 - Discretization
# ----------------------------------------------------

print("\n" + "=" * 75)
print("PART 2: DISCRETIZATION")
print("=" * 75)

numeric_columns = df.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nNumeric attributes:")
print(numeric_columns)

# Three bins as specified in the experiment
discretized_df = df.copy()

for column in numeric_columns:

    discretized_df[column] = pd.cut(
        discretized_df[column],
        bins=3,
        labels=[
            f"{column}_Low",
            f"{column}_Medium",
            f"{column}_High"
        ]
    )

print("\nDataset After Discretization:")
print(discretized_df.head())

# ----------------------------------------------------
# PART 3 - Apriori After Discretization
# ----------------------------------------------------

print("\n" + "=" * 75)
print("PART 3: APRIORI AFTER DISCRETIZATION")
print("=" * 75)

transactions_after = []

for _, row in discretized_df.iterrows():

    transaction = []

    for column in discretized_df.columns:

        value = row[column]

        if pd.notna(value):
            transaction.append(f"{column}={value}")

    transactions_after.append(transaction)

encoder_after = TransactionEncoder()

encoded_after = encoder_after.fit(
    transactions_after
).transform(transactions_after)

after_df = pd.DataFrame(
    encoded_after,
    columns=encoder_after.columns_
)

MIN_SUPPORT = 0.20
MIN_CONFIDENCE = 0.80

frequent_after = apriori(
    after_df,
    min_support=MIN_SUPPORT,
    use_colnames=True
)

print("\nMinimum Support:", MIN_SUPPORT)
print("Minimum Confidence:", MIN_CONFIDENCE)

if len(frequent_after) > 0:

    rules_after = association_rules(
        frequent_after,
        metric="confidence",
        min_threshold=MIN_CONFIDENCE
    )

    if len(rules_after) > 0:

        rules_after = rules_after.sort_values(
            by=["confidence", "lift"],
            ascending=False
        ).head(10)

        print("\nAssociation Rules After Discretization:")

        for _, rule in rules_after.iterrows():

            antecedent = ", ".join(
                sorted(list(rule["antecedents"]))
            )

            consequent = ", ".join(
                sorted(list(rule["consequents"]))
            )

            print(
                f"\n{antecedent} ==> {consequent}"
            )

            print(
                f"Support    = {rule['support']:.4f}"
            )

            print(
                f"Confidence = {rule['confidence']:.4f}"
            )

            print(
                f"Lift       = {rule['lift']:.4f}"
            )

    else:
        print("\nNo association rules found.")

else:
    print("\nNo frequent itemsets found.")

# ----------------------------------------------------
# Final observation
# ----------------------------------------------------

print("\n" + "=" * 75)
print("OBSERVATION")
print("=" * 75)

print(
    "Discretization converts continuous numeric attributes "
    "into categorical intervals."
)

print(
    "This reduces the number of unique values and makes "
    "association rules easier to interpret."
)

print("\n" + "=" * 75)
print("IRIS APRIORI EXPERIMENT COMPLETED")
print("=" * 75)
