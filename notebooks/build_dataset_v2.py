import pandas as pd

output_dir = r"C:\reposcore_data"

# Load batch 1 (already has readme_text)
batch1 = pd.read_csv(f"{output_dir}\\repos_with_readme.csv")

# Load batch 2 pieces and merge them
batch2_basic = pd.read_csv(f"{output_dir}\\repos_basic_batch2.csv")
batch2_readme = pd.read_csv(f"{output_dir}\\repos_enriched_batch2_with_readme.csv")

batch2 = batch2_basic.merge(
    batch2_readme[["full_name", "has_readme", "readme_size", "has_ci", "has_tests", "readme_text"]],
    on="full_name", how="inner"
)

# Combine both batches
df = pd.concat([batch1, batch2], ignore_index=True)

# Remove duplicates (in case any repo appeared in both batches)
df = df.drop_duplicates(subset="full_name")

# Drop rows with missing enrichment data
df = df.dropna(subset=["has_readme", "has_ci", "has_tests"])

# Recalculate derived features
df["created_at"] = pd.to_datetime(df["created_at"], format="mixed")
df["pushed_at"] = pd.to_datetime(df["pushed_at"], format="mixed")
now = pd.Timestamp.now(tz="UTC")
df["repo_age_days"] = (now - df["created_at"]).dt.days
df["days_since_last_commit"] = (now - df["pushed_at"]).dt.days
df["stars_per_month"] = df["stars"] / (df["repo_age_days"] / 30).clip(lower=1)

# Redefine quality label on the combined dataset
median_stars_per_month = df["stars_per_month"].median()
df["quality"] = (
    (df["stars_per_month"] > median_stars_per_month) &
    ((df["has_ci"] == True) | (df["has_tests"] == True))
).astype(int)

print("Total repos after merging and cleaning:", len(df))
print("Quality label distribution:")
print(df["quality"].value_counts())

df.to_csv(f"{output_dir}\\repos_combined_final.csv", index=False)
print(f"Saved combined dataset to {output_dir}\\repos_combined_final.csv")