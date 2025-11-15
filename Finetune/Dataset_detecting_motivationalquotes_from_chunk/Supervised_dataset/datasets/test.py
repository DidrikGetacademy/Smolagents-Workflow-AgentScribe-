import pandas as pd

# Load the parquet file
df = pd.read_parquet(r"c:\Users\didri\Downloads\train-00000-of-00001 (2).parquet")

# Convert to standard JSON (list of dicts)
output_file = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\datasets\filler_sentence.json"
df.to_json(output_file, orient="records", force_ascii=False, indent=2)

print(f"Saved {len(df)} records to {output_file}")
