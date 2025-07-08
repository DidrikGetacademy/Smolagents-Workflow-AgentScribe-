import json
import tiktoken

path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\train.jsonl"

# Velg encoding som passer modellen du skal bruke, f.eks. "cl100k_base" for GPT-4/3.5-turbo
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    tokens = encoding.encode(text)
    return len(tokens)

with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    if not lines:
        print("File is empty!")
    else:
        total_tokens = 0
        max_tokens = 0
        for i, line in enumerate(lines):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON on line {i+1}: {e}")
                break
            
            # Velg hvilke felter du teller tokens for — her teller vi både prompt og completion
            prompt = obj.get("prompt", "")
            completion = obj.get("completion", "")
            tokens_prompt = count_tokens(prompt)
            tokens_completion = count_tokens(completion)
            tokens_line = tokens_prompt + tokens_completion
            
            total_tokens += tokens_line
            if tokens_line > max_tokens:
                max_tokens = tokens_line
            
            print(f"Line {i+1}: prompt tokens = {tokens_prompt}, completion tokens = {tokens_completion}, total tokens = {tokens_line}")
        
        print(f"\nTotal lines: {len(lines)}")
        print(f"Max tokens in single line: {max_tokens}")
        print(f"Average tokens per line: {total_tokens / len(lines):.2f}")
