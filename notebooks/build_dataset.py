import pandas as pd

# Load both files
basic = pd.read_csv("data/repos_basic.csv")
enriched = pd.read_csv("data/repos_enriched_full.csv")

# Merge on full_name
df = basic.merge(enriched, on="full_name", how="inner")

# Drop rows where enrichment failed (connection errors -> None values)
df = df.dropna(subset=["has_readme", "has_ci", "has_tests"])

# Convert date columns to datetime
df["created_at"] = pd.to_datetime(df["created_at"])
df["pushed_at"] = pd.to_datetime(df["pushed_at"])

# Derived features
now = pd.Timestamp.now(tz="UTC")
df["repo_age_days"] = (now - df["created_at"]).dt.days
df["days_since_last_commit"] = (now - df["pushed_at"]).dt.days
df["stars_per_month"] = df["stars"] / (df["repo_age_days"] / 30).clip(lower=1)

# Define quality label
median_stars_per_month = df["stars_per_month"].median()
df["quality"] = (
    (df["stars_per_month"] > median_stars_per_month) &
    (df["has_ci"] == True) &
    (df["has_tests"] == True)
).astype(int)

print("Total repos after cleaning:", len(df))
print("Quality label distribution:")
print(df["quality"].value_counts())

df.to_csv("data/repos_final.csv", index=False)
print("Saved final dataset to data/repos_final.csv")