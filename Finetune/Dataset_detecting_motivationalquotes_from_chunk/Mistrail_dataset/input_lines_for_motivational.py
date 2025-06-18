import json
import re
# ✅ Correctly separated fields:
texts_to_insert = [
    {
        "tekst":
    """



        """,
        "quote": "",
        "Thought":  ""
    }




# 8️⃣ “People overestimate what they can do in a day and underestimate what they can do in a year.”
# (Variant from Bill Gates / Tony Robbins)

# 9️⃣ “Work in silence. Let success make the noise.”

# 🔟 “Nobody cares, work harder.”
# (Popular gym/discipline clip)

# 1️⃣1️⃣ “When you feel like giving up, remember why you started.”

# 1️⃣2️⃣ “Stars can’t shine without darkness.”

# 1️⃣3️⃣ “Fall in love with the process and the results will come.”

# 1️⃣4️⃣ “Be so good they can’t ignore you.”
# (Steve Martin quote, used in career TikToks)

# 1️⃣5️⃣ “A year from now, you’ll wish you started today.”

# 1️⃣6️⃣ “Small steps every day lead to big changes over time.”

# 1️⃣7️⃣ “Your comfort zone is killing your potential.”

# 1️⃣8️⃣ “You owe it to yourself to become everything you’ve ever dreamed of being.”

# 1️⃣9️⃣ “Success is rented. And the rent is due every day.”
# (Variant from motivational speakers like Rory Vaden)

# 2️⃣0️⃣ “One day, all your sacrifices will make sense.”
]

def insert_text_between_chunks(prompt_text, insert_text):
    pattern = r"(\[chunk start\])(.*?)(\[chunkend\])"
    return re.sub(pattern, rf"\1 {insert_text.strip()} \3", prompt_text, flags=re.DOTALL)

def insert_thought_in_completion(completion_text, thought_text):
    pattern = r"(Thought:\s*)(.*?)(\s*Code:)"
    replacement = rf"\1{thought_text.strip()} \3"
    return re.sub(pattern, replacement, completion_text, flags=re.DOTALL)

def insert_quote_in_completion(completion_text, quote_text):
    pattern = r'SaveMotivationalText\("(.*?)"\)'
    replacement = f'SaveMotivationalText("{quote_text.strip()}")'
    return re.sub(pattern, replacement, completion_text, flags=re.DOTALL)

def process_jsonl(input_file, output_file, texts):
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "a", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if i >= len(texts):
                print(f"Warning: No text for line {i+1}, skipping.")
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
