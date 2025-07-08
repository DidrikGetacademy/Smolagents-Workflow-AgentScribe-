import json

# File paths
json_file_path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\filler_sentence_dataset.json"
txt_file_path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\fillersentence.txt"

# Load existing JSON data
with open(json_file_path, 'r', encoding='utf-8') as jf:
    data = json.load(jf)

# Read new lines from the .txt file, split each in the middle, and append both halves
with open(txt_file_path, 'r', encoding='utf-8') as tf:
    for line in tf:
        sentence = line.strip()
        if sentence:  # Skip empty lines
            words = sentence.split()
            if len(words) > 10:
                mid = len(words) // 2
                first_half = ' '.join(words[:mid])
                second_half = ' '.join(words[mid:])
                data.append({"sentence": first_half})
                data.append({"sentence": second_half})
            else:
                data.append({"sentence": sentence})  # Keep 1-word lines as is

# Save the updated JSON back to file
with open(json_file_path, 'w', encoding='utf-8') as jf:
    json.dump(data, jf, indent=2, ensure_ascii=False)

print("✅ Sentences split and appended.")
