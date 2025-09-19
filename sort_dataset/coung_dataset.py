import pandas as pd

# Load the two datasets
train_df = pd.read_csv("/mnt/data/train_split.csv")
test_df = pd.read_csv("/mnt/data/test_split.csv")

def count_items(df):
    # Keep only relevant columns
    df = df[["category", "lab type", "unsafe"]].copy()
    
    # Map unsafe (0/1) to descriptive labels
    df["safety_label"] = df["unsafe"].map({0: "safe", 1: "unsafe"})
    
    # Ensure category is lowercase and split multiple categories
    df["category"] = df["category"].astype(str).str.lower()
    df_expanded = df.assign(
        category=df["category"].str.split(",")
    ).explode("category")
    
    # Remove any extra spaces
    df_expanded["category"] = df_expanded["category"].str.strip()
    
    # Group and count
    counts = (
        df_expanded
        .groupby(["lab type", "category", "safety_label"])
        .size()
        .reset_index(name="count")
    )
    
    return counts

# Run for both files
train_counts = count_items(train_df)
test_counts = count_items(test_df)

print(train_counts)
print(test_counts)
