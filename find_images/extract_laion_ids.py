import glob, json
import pyarrow.dataset as ds

# 1. Point to your .parquet files
parquet_files = glob.glob(r"D:\laion\*.parquet")
meta = ds.dataset(parquet_files, format="parquet")

# 2. (Optional) print schema to verify field names
print(meta.schema)
# ➜ SAMPLE_ID: int64
#   URL:       string
#   …  

# 3. Load your URL set
with open("filtered_url.txt") as f:
    target = set(line.strip() for line in f)

# 4. Pull a filtered Table with the exact column names
table = meta.to_table(
    filter=ds.field("URL").isin(target),          # uppercase URL
    columns=["URL", "SAMPLE_ID"]                  # uppercase, and SAMPLE_ID instead of laion_id
)

# 5. Build your mapping
mapping = dict(zip(
    table.column("URL").to_pylist(),
    table.column("SAMPLE_ID").to_pylist()
))

# 6. Save to JSON
with open("url_to_id.json", "w") as f:
    json.dump(mapping, f)
