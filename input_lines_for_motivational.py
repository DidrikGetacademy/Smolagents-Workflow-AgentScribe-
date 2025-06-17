import json
import re

# Liste med tekst og quote per rad
texts_to_insert = [
    {
        "tekst": """[2025.30s - 2030.55s] I know some days it feels like nothing’s moving forward. Like you’re stuck in a loop of frustration and doubt.  \\\n[2030.56s - 2036.10s] But progress isn’t always loud or flashy. Sometimes it’s quiet, almost invisible.  \\\n[2036.10s - 2042.77s] It’s the small steps you take when no one’s watching. The discipline to keep going when motivation fades.  \\\n[2042.78s - 2049.34s] “Success is often found in the moments when you choose persistence over comfort.”  \\\n[2049.35s - 2055.22s] That one thought changed how I approach my goals—it’s a reminder that what matters most is the effort you put in consistently.  \\\n[2055.23s - 2061.90s] So even if today feels like a failure, or a pause, it’s actually part of the process. You’re building something strong underneath""",
        "quote": "[2042.78s - 2049.34s]Success is often found in the moments when you choose persistence over comfort"
        "Thought" """   """
    },
    {
 
    },
]

def insert_text_between_chunks(prompt_text, insert_text):
    """Setter inn tekst mellom [chunk start] og [chunk end]."""
    insert_text_single_line = insert_text.strip()
    pattern_full = r"(\[chunk start\])(.*?)(\[chunk end\])"
    if re.search(pattern_full, prompt_text, flags=re.DOTALL):
        new_prompt = re.sub(pattern_full, rf"\1 {insert_text_single_line} \3", prompt_text, flags=re.DOTALL)
    else:
        pattern_start_only = r"(\[chunk start\])(.*)"
        new_prompt = re.sub(pattern_start_only, rf"\1 {insert_text_single_line}", prompt_text, flags=re.DOTALL)
    return new_prompt

def insert_quote_in_completion(completion_text, quote_text):
    """Setter inn quote i SaveMotivationalText("...")."""
    pattern = r'SaveMotivationalText\(".*?"\)'
    new_completion = re.sub(pattern, f'SaveMotivationalText("{quote_text}")', completion_text, flags=re.DOTALL)
    return new_completion

def insert_thought_in_completion(completion_text, thought_text):
    """Setter inn Thought: <tekst> rett etter Thought: ."""
    pattern= r'Thought:\s*.*?(?=\\nCode)'
    replacement = f'Thought: {thought_text}'
    new_completion =  re.sub(pattern,replacement,completion_text, flags=re.DOTALL)
    return new_completion
    

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
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                continue
            data = texts[i]
            tekst = data.get("tekst", "")
            quote = data.get("quote", "")
            thought = data.get("Thought", "")


            if "prompt" in obj:
                obj["prompt"] = insert_text_between_chunks(obj["prompt"], tekst)
            if "completion" in obj:
                obj["completion"] = insert_quote_in_completion(obj["completion"], quote)
                obj["completion"] = insert_thought_in_completion(obj["completion"], thought)

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    input_path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\input_savemotivationaltext.jsonl"
    output_path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\output.jsonl"
    process_jsonl(input_path, output_path, texts_to_insert)
