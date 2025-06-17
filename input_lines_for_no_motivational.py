import json
import re

# Liste med tekster, én per linje i input.jsonl
texts_to_insert = [
    """ 


    """,
]

def insert_text_between_chunks(prompt_text, insert_text):
    insert_text_single_line = insert_text.strip()
    pattern_full = r"(\[chunk start\])(.*?)(\[chunk end\])"
    if re.search(pattern_full, prompt_text, flags=re.DOTALL):
        new_prompt = re.sub(pattern_full, rf"\1 {insert_text_single_line} \3", prompt_text, flags=re.DOTALL)
    else:
        pattern_start_only = r"(\[chunk start\])(.*)"
        new_prompt = re.sub(pattern_start_only, rf"\1 {insert_text_single_line}", prompt_text, flags=re.DOTALL)
    return new_prompt

def process_jsonl(input_file, output_file, texts):
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "a", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if i >= len(texts):
                print(f"Advarsel: Ingen tekst å sette inn for linje {i+1}, hopper over.")
                fout.write(line)
                continue
            if "prompt" in obj:
                obj["prompt"] = insert_text_between_chunks(obj["prompt"], texts[i])
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    input_path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\input_no_text_save.jsonl"
    output_path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\output.jsonl"
    process_jsonl(input_path, output_path, texts_to_insert)
