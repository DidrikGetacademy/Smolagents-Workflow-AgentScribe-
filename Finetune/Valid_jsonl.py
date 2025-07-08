import json

path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Mistrail_dataset\mistrail_finetune.jsonl"

with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    if not lines:
        print("File is empty!")
    else:
        valid_lines = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                # Skip empty lines silently
                continue
            try:
                obj = json.loads(line)
                valid_lines += 1
            except json.JSONDecodeError as e:
                print(f"Invalid JSON on line {i+1}: {e}")
        print(f"File has {valid_lines} valid JSON lines.")
