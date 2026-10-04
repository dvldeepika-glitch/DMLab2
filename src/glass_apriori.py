import pandas as pd
import numpy as np
from scipy.io import arff
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

print("=" * 75)
print("DATA MINING LAB 2 - GLASS APRIORI")
print("=" * 75)

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------
file_path = "datasets/glass.arff"

data, meta = arff.loadarff(file_path)
df = pd.DataFrame(data)

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )

print("\nDataset: glass.arff")
print("Original Instances:", len(df))
print("Attributes:", len(df.columns))

print("\nOriginal Dataset:")
print(df.head())

# ------------------------------------------------------------
# 2. RESAMPLE - 50%
# ------------------------------------------------------------
print("\n" + "=" * 75)
print("PART 1: RESAMPLING")
print("=" * 75)

sample_df = df.sample(
    frac=0.50,
    random_state=42
).reset_index(drop=True)

print("Sample Size: 50%")
print("Original Instances:", len(df))
print("Resampled Instances:", len(sample_df))

print("\nResampled Dataset:")
print(sample_df.head())

# ------------------------------------------------------------
# 3. DISCRETIZATION
# ------------------------------------------------------------
print("\n" + "=" * 75)
print("PART 2: DISCRETIZATION")
print("=" * 75)

numeric_columns = sample_df.select_dtypes(
    include=[np.number]
).columns.tolist()

print("Numeric attributes:")
print(numeric_columns)

for col in numeric_columns:

    # Get number of unique values
    unique_count = sample_df[col].nunique()

    if unique_count >= 3:

        # Create 3 equal-frequency bins
        sample_df[col] = pd.qcut(
            sample_df[col],
            q=3,
            labels=False,
            duplicates="drop"
        )

        # Convert bin numbers to names
        sample_df[col] = sample_df[col].map({
            0: "Low",
            1: "Medium",
            2: "High"
        })

    else:
        # If attribute has fewer than 3 unique values,
        # keep its values as categorical
        sample_df[col] = sample_df[col].astype(str)

print("\nDataset After Discretization:")
print(sample_df.head())

# ------------------------------------------------------------
# 4. CREATE TRANSACTIONS
# ------------------------------------------------------------
transactions = []

for _, row in sample_df.iterrows():

    transaction = []

    for col in sample_df.columns:
        transaction.append(
            str(col) + "=" + str(row[col])
        )

    transactions.append(transaction)

# ------------------------------------------------------------
# 5. TRANSACTION ENCODING
# ------------------------------------------------------------
te = TransactionEncoder()

te_array = te.fit(
    transactions
).transform(transactions)

transaction_df = pd.DataFrame(
    te_array,
    columns=te.columns_
)

# ------------------------------------------------------------
# 6. APRIORI
# ------------------------------------------------------------
print("\n" + "=" * 75)
print("PART 3: APRIORI ASSOCIATION RULE MINING")
print("=" * 75)

min_support = 0.10
min_confidence = 0.70

print("Minimum Support   :", min_support)
print("Minimum Confidence:", min_confidence)

frequent_itemsets = apriori(
    transaction_df,
    min_support=min_support,
    use_colnames=True
)

print("\nNumber of Frequent Itemsets:",
      len(frequent_itemsets))

if len(frequent_itemsets) > 0:

    print("\nFrequent Itemsets:")
    print(frequent_itemsets)

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    rules = rules.sort_values(
        by=["confidence", "lift"],
        ascending=False
    )

    print("\n" + "=" * 75)
    print("TOP ASSOCIATION RULES")
    print("=" * 75)

    if len(rules) == 0:

        print("No association rules satisfy the confidence threshold.")

    else:

        for number, (_, row) in enumerate(
            rules.head(10).iterrows(),
            start=1
        ):

            antecedent = ", ".join(
                sorted(list(row["antecedents"]))
            )

            consequent = ", ".join(
                sorted(list(row["consequents"]))
            )

            print(
                f"\nRule {number}: "
                f"{antecedent} ==> {consequent}"
            )

            print(
                f"Support    = {row['support']:.4f}"
            )

            print(
                f"Confidence = {row['confidence']:.4f}"
            )

            print(
                f"Lift       = {row['lift']:.4f}"
            )

else:

    print("No frequent itemsets found.")

# ------------------------------------------------------------
# 7. OBSERVATION
# ------------------------------------------------------------
print("\n" + "=" * 75)
print("OBSERVATION")
print("=" * 75)

print("""
Resampling reduces the Glass dataset to 50% of its
original size.

Discretization converts continuous numeric attributes
into categorical intervals such as Low, Medium and High.

The preprocessing steps make the continuous Glass data
more suitable for Apriori association rule mining.

Resampling reduces the number of transactions, while
discretization improves the interpretability of rules.
""")

print("=" * 75)
print("GLASS APRIORI EXPERIMENT COMPLETED")
print("=" * 75)