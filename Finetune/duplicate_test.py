import json

# Load the dataset
with open(r'C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\datasets\filler_sentence_dataset.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

quotes_seen = set()
unique_data = []
duplicates_count = 0

for entry in data:
    quote = entry['sentence']
    if quote not in quotes_seen:
        unique_data.append(entry)
        quotes_seen.add(quote)
    else:
        duplicates_count += 1

total_before = len(data)
total_after = len(unique_data)

print(f"Total quotes before deduplication: {total_before}")
print(f"Duplicates found and removed: {duplicates_count}")
print(f"Total quotes after deduplication: {total_after}")

# Save unique quotes to a new file
with open(r'C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\dada.json', 'w', encoding='utf-8') as file:
    json.dump(unique_data, file, ensure_ascii=False, indent=4)
