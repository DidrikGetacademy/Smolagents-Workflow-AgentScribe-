import json

path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Mistrail_dataset\mistrail_finetune.jsonl"

with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    if not lines:
        print("File is empty!")
    else:
        for i, line in enumerate(lines):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON on line {i+1}: {e}")
                break
        else:
            print(f"File has {len(lines)} valid JSON lines.")