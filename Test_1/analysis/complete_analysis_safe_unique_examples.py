import pandas as pd

count_file  = "count_sample_space_auto.csv"
safe_file   = "no_crashes.csv"
output_file = "count_sample_space_auto_with_safe.csv"

# Read the files
count_df = pd.read_csv(count_file)
safe_df  = pd.read_csv(safe_file)

print(f"Count df rows: {len(count_df):,}")
print(f"Safe df rows : {len(safe_df):,}\n")

# Define the columns that identify an "example" (all columns except count and index)
example_cols = ['action', 'curr_lane', 'free_E', 'free_NE', 'free_NW', 'free_SE', 'free_SW', 'free_W']
print(f"Checking match on columns: {example_cols}")

# Create a set of all safe examples (tuples of the identifying columns)
safe_examples = set(tuple(row) for row in safe_df[example_cols].itertuples(index=False, name=None))

# Check each row in count_df: is this example in the safe set?
def check_if_safe(row):
    example = tuple(row[example_cols])
    return "True" if example in safe_examples else "False"

count_df["safe"] = count_df.apply(check_if_safe, axis=1)

# Show results
print("\nResults:")
print(count_df["safe"].value_counts())
print("\nPercentages:")
print(count_df["safe"].value_counts(normalize=True).round(3) * 100, "%")

# Optional: see breakdown by action
print("\nSafe rate by action:")
print(count_df.groupby("action")["safe"].value_counts().unstack().fillna(0))

# Save the result
count_df.to_csv(output_file, index=False)
print(f"\nSaved → {output_file}")
