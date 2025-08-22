import json
import random
import re
from datasets import load_dataset
from transformers import AutoTokenizer
import re

CONFIG = {
    'model_name': r'C:\Users\didri\Desktop\LLM-models\LLM-Models\microsoft\unsloth\Phi-4-mini-instruct',
}
count_only_filler = 0
count_single_Quote_mix = 0
count_single_quote_natural = 0
Count_double_quotes = 0
def log(msg):
    print(msg)

def add_timestamps_and_filler(sentences, filler_before, filler_after, base_start=random.uniform(300.0, 1200.0), avg_sentence_duration=4.0):
    all_sents = filler_before + sentences + filler_after
    timestamped_text = ""
    current_time = base_start
    for sent in all_sents:
        duration = random.uniform(avg_sentence_duration * 0.8, avg_sentence_duration * 1.2)
        start_time = current_time
        end_time = current_time + duration
        timestamped_text += f"[{start_time:.2f}s - {end_time:.2f}s] {sent.strip()}\n"
        current_time = end_time + 0.1
    return timestamped_text

def split_quote_randomly(quote, max_words_per_line=10):
    words = quote.strip().split()
    chunks = []
    i = 0
    while i < len(words):
        n = random.randint(4, max_words_per_line)
        chunk = words[i:i+n]
        if chunk:
            chunks.append(" ".join(chunk))
        i += n
    return chunks

def save_as_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_filler_sentences():
    path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\datasets\filler_sentence_dataset.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        filler_sentences = [entry["sentence"].strip() for entry in data if 10 < len(entry["sentence"].strip()) < 150]
    log(f"[Dataset Load] Loaded {len(filler_sentences)} filler sentences from local JSON")
    return filler_sentences


def extract_quote_only_lines(quote_start_line, quote_lines, n_quote_start):
    # Del opp første linje i ord
    match = re.match(r"\[(.*?)s - (.*?)s\] (.+)", quote_start_line)
    if not match:
        return []
    start_t, end_t, line_text = match.groups()
    words = line_text.strip().split()
    quote_words = words[-n_quote_start:]  # De siste ordene er quote
    quote_text = " ".join(quote_words)
    quote_start_clean = f"[{float(start_t):.2f}s - {float(end_t):.2f}s] {quote_text}"

    return [quote_start_clean] + quote_lines

def sample_unique_fillers(n, filler_pool, used_set):
    available = list(set(filler_pool) - used_set)
    if not available:
        return []
    selected = random.sample(available, min(n, len(available)))
    used_set.update(selected)
    return selected




def preprocess_with_filler_balanced(example, idx, filler_sentences, used_filler_sentences, tokenizer, dataset_quotes=None):
    global count_only_filler
    global count_single_Quote_mix
    global count_single_quote_natural
    global Count_double_quotes
    system_prompt = """
You are an attentive assistant who reads everything carefully. Your primary goal is to identify self-contained text for creating motivational shorts by identifying short-form motivational or self-improvement statements and anecdotes that inspire personal growth, resilience, and self-reflection — often using contrasts, memorable insights, or real-life stories to encourage positive mindset, discipline, and perseverance. Your task is to carefully analyze and read a timestamped chunk and extract any Qualifying motivational texts, inspirational passages, or self-contained statements that offer encouragement, life advice, or inspiration and could be used for a 15 - 20 seconds short video. If such texts are found, save them using the appropriate function. If none are found, return a final response indicating that.
You are to save standalone Qualifying motivational texts, advice, inspiring messages, or passages that are complete and do not lack context or would confuse a listener if isolated from the rest of the chunk.
Analyze the chunk/transcript between [chunk start] and [chunk end].
Objective: Your job is to extract motivational qualifying texts from the input text chunk/transcript. 
These are typically short, self-contained passages that offer encouragement, life advice, or inspiration — including inspiring messages, anecdotes, or insights (not limited to direct quotes).
Reasoning: Always begin with a Thought: statement explaining your reasoning — for example, summarize the intent of the chunk, indicate whether you identified any motivational texts (and how many), or explain why none qualified.
ABSOLUTE OUTPUT FORMAT RULE
YOU must output exactly:
    Thought: [your reasoning text here]  
    <code>  
    [one or more SaveMotivationalText(...) calls if any found]  
    [exactly one final_answer(...) call]  
    </code>

Your output format rules:
    1. Thought: must always be outside <code>.
    2. <code> must contain nothing except function calls.
    3. Always provide a 'Thought:' sequence, and a '<code>' sequence ending with '</code>', else you will fail.
    4. The Thought: sequence must always provide a short reasoning over the overall intent of the chunk, describing what the speaker/text is mainly doing (e.g., telling a story, joking, giving advice, reflecting, describing an event). Then, based on that intent, explain whether any qualifying motivational texts were found or not. If qualifying texts were found, explain briefly why those specific passages qualify (mention their motivational themes like effort, discipline, resilience, growth, mindset shift, etc.), why other parts of the chunk were excluded (e.g., anecdotal, casual, off-topic), and explicitly confirm that the saved motivational texts are complete, self-contained, and do not lack context when isolated. If no qualifying texts were found, explain that the chunk’s intent (story, casual chat, filler, incomplete thought, etc.) did not contain any standalone motivational passages suitable for a short video.

After analyzing chunk:
    -If one or more qualifying motivational texts are found: Output one SaveMotivationalText(...) call for each self-contained qualifying text.
    -After the last SaveMotivationalText(...) call, output exactly one final_answer("im done").
    -If none qualifying motivational texts  are found: 
        -Output only one line: final_answer("After carefully analysing the chunk/text, i have concluded nothing can be saved. Nothing qualifies for a motivational shorts video, No text had a clear beginning & end or a profound overall intent/meaning that is suitable for a motivational shorts. Nothing that would provide any interest for a listener.")

Here are 3 examples of correct output:
Exsample 1.  If two qualifying motivational texts are found in the chunk you analyze, You must output:
Thought: [Your reasoning here...]
<code>
    SaveMotivationalText(text="[623.70s - 627.11s] The magic you are looking for [627.11s - 640.14s]  is in the work you are avoiding", text_file=text_file)  
    SaveMotivationalText(text="[500.00s - 502.34s] You don't need perfect conditions to make progress. [502.34s - 505.22s] You just need to move", text_file=text_file)    
    final_answer("im done")
</code>

Exsample 2.  If one qualifying motivational text is found in the chunk/transcript that you analyze, You must output:
Thought: [your reasoning here...]
<code>
    SaveMotivationalText(text="[617.70s - 627.11s] You will encounter many challenges in life  [627.12s - 628.00s] But you must never be defeated by [628.01s - 629.55s] the challenges", text_file=text_file)
    final_answer("im done")
</code>    

Exsample 3. If no qualifying motivational texts are found in the chunk you analyze, You should output the reason in the The 'Thought' sequence, a short reason of the intent and noting that that the chunk (inferred as non-motivational or lacks a clear intent):
Thought: [your reasoning here...]
<code>
    final_answer("After carefully analysing the chunk/text, i have concluded nothing can be saved. Nothing qualifies for a motivational shorts video, That would grab the attention of a listener")
</code>

Timestamp Handling when saving qualifying text:
- When a motivational text spans multiple lines (each line containing a separate timestamp):
- Merge the lines into a single text.
- Include the start time from the first line and all timestamps up to and including the end time from the last line, Example: SaveMotivationalText(text="[617.70s - 627.11s] You will encounter many challenges in life [627.12s - 628.00s] But you must never be defeated by [628.01s - 629.55s] the challenges", text_file=text_file).
- Preserve original spacing and punctuation exactly.
- Example output: SaveMotivationalText("[start - end] Qualifying text line 1 [start - end] Qualifying text line 2 [start - end]", text_file=text_file) if the qualifying text spans multiple lines.
- The timestamps in the format [SSSS.SSs - SSSS.SSs] represent the time in seconds for when the words are spoken in the video transcript.

Here you have Important Rules/instructions to follow:
- Motivational texts must be inspirational and standalone — avoid fragments or generic sentences.
- Always include both `Thought:` and `<code>` blocks.
- Use exact function names and punctuation as shown.
- Do not return texts that are incomplete or unclear.
- Do not create multiple SaveMotivationalText() calls for each line in a single text.
- Do not alter or guess missing timestamps — use the exact start and end values provided in the lines that contain the text.
- Text should appear as a single, continuous string, even if it was originally split across 1-8 lines.   
- Each line in a chunk has a timestamp ([start - end]) that represent the time of those words spoken, the transcript you are analyzing is text from a video transcribed from a audio.
- The chunks you analyze may vary in size. Always analyze the entire chunk and identify any qualifying text if any before providing <code>.

Types of Qualifying Motivational Texts:
- Passages that encourage perseverance, personal growth, resilience, mindset shift, success, discipline, consistency, or overcoming challenges (e.g., "You will encounter many challenges in life, but you must never be defeated by them"). A complete text that does not lack context when isolated.
- Action-oriented advice that inspires immediate steps toward improvement (e.g., "You don’t need perfect conditions to make progress. You just need to move").
- Messages promoting self-belief, confidence, or personal growth (e.g., "The difference between who you are and who you want to be is in the choices you make every day").
- Universal life advice that is concise and impactful (e.g., "Discipline is choosing what you want most over what you want now").
- Inspirational statements that evoke hope or determination (e.g., "Fear is loud, but your future deserves to be louder").
- Anecdotes or stories that illustrate growth, resilience, or lessons (e.g., short, self-contained real-life examples from temp.txt-like files).

Notes on Qualifying Texts:
- Texts must be self-contained, meaning they convey a complete thought without needing surrounding context.
- Avoid generic statements (e.g., "Life is hard" or "mindset is good") or fragments that lack clear motivational intent and would not grab a listener's attention.
- Ensure the passage is concise enough for a motivational shorts video.
- Do not save text that does not form a complete thought. If it lacks context when isolated from the rest of the transcript, do not save it. You must understand the overall meaning of the text to judge whether it stands on its own. If it's not a self-contained motivational passage, you will fail.

Here are 5 few-shot examples of qualifying texts, and If you identify similar texts like these that qualifies in a transcript or chunk that you analyze, save it with SaveMotivationalText("...",text_file=text_file) as they are self-contained and suitable for  motivational shorts video:
    -----------------
    1. "always keep going there's been times particularly early in my career where it just feels like thisis the end but what i've come to find out is that no matter what happens the stormeventually ends and when the storm does end you want to make sure that you're ready"
        Reason: Qualifies because it’s a self-contained motivational lesson about perseverance and resilience. It conveys a complete thought that encourages hope and preparation.


    2. "James Clear has this fucking unbelievable insight. It doesn't make sense to continue wanting something if you're not willing to do what it takes to get it. If you don't want to live the lifestyle, then release yourself from the desire. To crave the result, but not the process, is to guarantee disappointment"
        Reason: Qualifies because it offers a profound motivational insight with complete context. It delivers actionable advice on aligning desires with actions, making it suitable as a standalone short.
    
    3. "The magic you are looking for is in the work you are avoiding"
         Reason: Qualifies because it’s concise, memorable, and self-contained. It provides clear motivational advice about discipline and effort without requiring extra context.

    4. "Willpower is the key to success. Successful people strive no matter what they feel by applying their will to overcome apathy, doubt or fear."
        Reason:  Qualifies because it presents a universal motivational principle about resilience and determination. It is self-contained and inspires immediate application.
    
    5. "Discipline isn't just about following rigid rules or punishing yourself for slip-ups; it's the bridge between your dreams and reality, where every small, consistent action you take today—like waking up early to work on your goals despite feeling tired—compounds into massive personal growth tomorrow, turning potential into achievement and weakness into unshakeable strength"
        Reason: Qualifies because it’s a full motivational passage that explains discipline in a clear, actionable, and inspirational way. It has a strong beginning, middle, and end, making it suitable as a complete short.
    
Here are 5 few-shot examples of Texts that does not Qualify and that you must never save with SaveMotivationalText("...",text_file=text_file) Even if they appear motivational at first glance, they must be excluded. Below the texts is a (reason) that explain why the text does not qualify. Thought: must reflect why they fail to qualify: these texts are either incomplete, generic, dependent on prior context, or too vague to stand on their own as a motivational short. Saving them would confuse or disengage a listener because they lack a clear beginning, end, or a profound intent.
    -----------------
    1. "Life can be tough sometimes" 
      Reason: excluded because Not qualifying text because Too generic, lacks insight or actionable advice, not engaging enough for a short video.

    2. "And that's why you should keep trying" 
        Reason: excluded because Depends on missing context (what was discussed before), incomplete if isolated.

    3. "Like we discussed before, success is important" 
        Reason: excluded because References prior discussion, cannot stand alone, vague and redundant.

    4. "I got to push you back. I got to stop you from going that hard." 
        Reason: excluded because Fragmented statement, lacks motivational theme or universal meaning, unclear intent.
        
    5. "But I said, at this point, I think I've been working out like crazy person" 
        Reason: excluded because Personal anecdote without a motivational lesson, lacks self-contained insight.
"""
   
    instruction = """
        Your task is to Identify Qualifying Motivational Texts & Save them if any is found in the chunk.
        Here is the chunk you must analyze: \n
    """

    if random.random() < 0.3:
        count_only_filler += 1
        n_fillers = random.randint(4, 10)
        selected_fillers = sample_unique_fillers(n_fillers, filler_sentences, used_filler_sentences)
        if selected_fillers is None:
            return None


        chunk_text = "[chunk start]\n" + add_timestamps_and_filler([], selected_fillers, []) + "[chunk end]\n"
        label_text = (
            'Thought:  \n' +
            '<code>\n' +
            'final_answer("After carefully analysing the chunk/text, i have concluded nothing can be saved. Nothing qualifies for a motivational shorts video, That would grab the attention of a listener")\n' +
            '</code>' 
        )

    else:
        quote_filler_mix_bool = False
        quote_filler_mix_chance = 0.5
        double_quote_chance = 0.2
  
        if dataset_quotes is None:
            dataset_quotes = [example["text"]]

        def clean_quote(q):
            return q.replace("“", '').replace("”", '').replace("‘", "").replace("’", "").replace(".", "").replace('""', '').strip()

        if random.random() < double_quote_chance and len(dataset_quotes) > 1:
            Count_double_quotes += 1
            other_example = random.choice(dataset_quotes)
            while other_example == example["quote"]:
                other_example = random.choice(dataset_quotes)
            quotes = [clean_quote(example["quote"]), clean_quote(other_example)]

        else:
            quotes = [clean_quote(example["quote"])]

        all_sentences = []
        quotes_splits = []
        quote_filler_mix_bool = (len(quotes) == 1) and (random.random() < quote_filler_mix_chance)
        for i, q in enumerate(quotes):
            random_max_words = random.randint(4,7)

            # Split quote i mindre linjer som vanlig
            quote_sents = split_quote_randomly(q, max_words_per_line=random_max_words)
            print(f"Spltted quotes randomaly: {quote_sents}")

            if quote_filler_mix_bool and len(quotes) == 1:
                count_single_Quote_mix += 1
                #henter ut en filler linje
                filler_intro = sample_unique_fillers(1, filler_sentences, used_filler_sentences)
                if filler_intro is not None and len(filler_intro) > 0: # hvis filler_intro ikke er null og mengden filler_intro er større enn 0
                  #  start_end_mix_chance = 0.5
                   # if random.random() > start_end_mix_chance:
                        
                    quote_words = quote_sents[0].split() # splitter en quote linje fra første linja/starten av quoten i lista til ord.
                    quote_word_amount = random.randint(1, len(quote_words))
                        
                    moved_words = " ".join(quote_words[:quote_word_amount])
                    remaining_quote = " ".join(quote_words[quote_word_amount:])

                    #slå sammen filler og første quote-linje
                    randomkChance = 0.5
                    #only_start = 0.5
                    #if random.random() < only_start:

                    if random.random() < randomkChance:

                        mixed_line = filler_intro[0].rstrip(".") + ", " + moved_words
                    else: 
                        mixed_line = filler_intro[0].rstrip(".") + " " + moved_words
                 #   else:
                   #     mixed_line_end = filler_intro[-1].rstrip(".") +  " " + moved_words
                    if remaining_quote:
                        quote_sents[0] = mixed_line
                        #quote_sents[-1] = mixed_line_end
                        quote_sents.insert(1, remaining_quote)
                    else:
                        quote_sents[0] = mixed_line
                    # else:
                    #     quote_words = quote_sents[-1].split() 
                    #     quote_word_amount = random.randint(1,len(quote_words) - 2)

                    #     moved_words = " ".join(quote_words[-quote_word_amount:])
                    #     remaining_quote = " ".join(quote_words[:-quote_word_amount])

                    #     randomkChance = 0.5
                    #     if random.random() < randomkChance:
                    #         mixed_filler = moved_words + ", " + filler_intro[0].lstrip().capitalize()
                    #     else:
                    #         mixed_filler = moved_words + " " + filler_intro[0].lstrip().capitalize()

                    #     if remaining_quote:
                    #         quote_sents[-1] = remaining_quote
                    #         filler_intro[0] = mixed_filler
                    #     else:
                    #         quote_sents.pop()  
                    #         filler_intro[0] = mixed_filler



            elif quote_filler_mix_bool == False and len(quotes) == 1:
                count_single_quote_natural += 1
            


            # legg til quote linjer i listen
            quotes_splits.append(quote_sents)
            all_sentences.extend(quote_sents)
            


            #Vis det er flere enn en quote legg til random fillers mellom dem
            if i < len(quotes) - 1:
                n_fillers_mid = random.randint(1, 2)
                fillers_mid = sample_unique_fillers(n_fillers_mid, filler_sentences, used_filler_sentences)
                if fillers_mid is None:
                    return None
                all_sentences.extend(fillers_mid)

        filler_before = sample_unique_fillers(random.randint(1, 3), filler_sentences, used_filler_sentences)
        if filler_before is None or len(filler_before) == 0:
            log(f"[Warning] Ran out of filler sentences for filler_before at idx {idx}")

            return None
        filler_after = sample_unique_fillers(random.randint(1, 3), filler_sentences, used_filler_sentences)
        if filler_after is None or len(filler_after) == 0:
            log(f"[Warning] Ran out of filler sentences for filler_before at idx {idx}")
            return None
        
        chunk_text = add_timestamps_and_filler(all_sentences, filler_before, filler_after)
        chunk_text = "[chunk start]\n" + chunk_text + "[chunk end]\n"

        chunk_lines = [line.strip() for line in chunk_text.strip().splitlines() if line.strip()]
        timestamp_re = re.compile(r"\[(\d+\.\d+)s - (\d+\.\d+)s\] (.+)")
        parsed_lines = []
        for line in chunk_lines:
            m = timestamp_re.match(line)
            if m:
                start_t = float(m.group(1))
                end_t = float(m.group(2))
                text = m.group(3).strip()
                parsed_lines.append((start_t, end_t, text))

        flat_quote_lines = []
        for quote_chunk in quotes_splits:
            flat_quote_lines.extend(quote_chunk)

        matched_quote_lines = []
        pi = 0
        for q_line in flat_quote_lines:
            found = False
            q_line_clean = q_line.strip().lower()
            while pi < len(parsed_lines):
                start_t, end_t, line_text = parsed_lines[pi]
                line_text_clean = line_text.strip().lower()
                if q_line_clean in line_text_clean or line_text_clean in q_line_clean:
                    matched_quote_lines.append((start_t, end_t, line_text))
                    pi += 1
                    found = True
                    break
                pi += 1
            if not found:
                matched_quote_lines.append((None, None, q_line))

      
        if len(quotes) == 1 and quote_filler_mix_bool:
            parts = []
            print(f"quotes: {quotes}\n")
            print(f"matched_quote_lines: {matched_quote_lines}\n")
            quote = quotes[0]
            print(f"Quote: {quote}\n")
            quote_words = quote.split()  # splitt sitatet i ord
            print(f"Quote words splitted: {quote_words}\n")
            
            # Rens sitatordene for punktum, komma osv. og gjør lowercase for sammenligning
            quote_words_cleaned = [re.sub(r"[^\w\s’']", '', word).lower() for word in quote_words]
            print(f"Quote words cleaned: {quote_words_cleaned}\n")

            i = 0  # indeks for sitatord
            new_matched_quote_lines = []

            for start_t, end_t, text in matched_quote_lines:
                original_words = text.strip().split()
                cleaned_words = [re.sub(r"[^\w\s’']", '', w).lower() for w in original_words]

                matched_words = []
                for orig_word, clean_word in zip(original_words, cleaned_words):
                    if i < len(quote_words_cleaned) and clean_word == quote_words_cleaned[i]:
                        # Ta med ordet fra sitatet med original casing (fra quote_words)
                        matched_words.append(quote_words[i])
                        i += 1
                    else:
                        # Ord matcher ikke neste ord i sitatet, hopp over det
                        pass
                
                if matched_words:
                    new_matched_quote_lines.append((start_t, end_t, ' '.join(matched_words)))
                
                if i >= len(quote_words_cleaned):
                    # Hele sitatet er funnet, stopp
                    break

            # Bygg opp output med tidsstempler og teksten som matcher
            for start_t, end_t, text in new_matched_quote_lines:
                if start_t is not None and end_t is not None:
                    parts.append(f"[{start_t:.2f}s - {end_t:.2f}s] {text}")
                else:
                    parts.append(text)

            full_quote_with_timestamps = " ".join(parts).replace('"', '\\"')
            print(f"parts: {parts}")
            print(f"full quote with timestamps: {full_quote_with_timestamps}")

            label_text = (
                'Thought: \n' +
                '<code>\n' +
               f'SaveMotivationalText(text="{full_quote_with_timestamps}", text_file=text_file)\n' +
                'final_answer("im done")\n' +
                '</code>'
            )
            

        elif len(quotes) == 1 and not quote_filler_mix_bool:
            parts = []
            for start_t, end_t, text in matched_quote_lines:
                if start_t is not None and end_t is not None:
                    parts.append(f"[{start_t:.2f}s - {end_t:.2f}s] {text}")
                else:
                    parts.append(text)
            full_quote_with_timestamps = " ".join(parts).replace('"', '\\"')
            label_text = (
                'Thought: \n' +
                '<code>\n' +
                f'SaveMotivationalText(text="{full_quote_with_timestamps}", text_file=text_file)\n' +
                'final_answer("im done")\n'
                '</code>'
            )

        else:
            save_calls = []
            idx_line = 0
            for quote_chunk in quotes_splits:
                parts = []
                for _ in quote_chunk:
                    start_t, end_t, text = matched_quote_lines[idx_line]
                    idx_line += 1
                    if start_t is not None and end_t is not None:
                        parts.append(f"[{start_t:.2f}s - {end_t:.2f}s] {text}")
                    else:
                        parts.append(text)
                combined_text = " ".join(parts).replace('"', '\\"')
                save_calls.append(f'SaveMotivationalText(text="{combined_text}", text_file=text_file)')
            Thought = f""

            
            label_text = (
                f"Thought: {Thought}\n" +
                "<code>\n" + "\n".join(save_calls) +
                'final_answer("im done")\n' +
                '</code>'
            )
            

    return {
        "messages": [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": instruction.strip() + "\n" + chunk_text.strip()},
            {"role": "assistant", "content": label_text.strip()}
        ]
    }
def load_quote_dataset():
    path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\datasets\Quote_dataset.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    quotes = [entry["quote"].strip() for entry in data if entry["quote"].strip()]
    return quotes

def main():
    used_filler_sentences = set()

    filler_sentences = load_filler_sentences()
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'], trust_remote_code=True)

    quotes_list = load_quote_dataset()
    dataset = [{"quote": q} for q in quotes_list]



    quotes_list = [x['quote'] for x in dataset]

    train_data = []
    train_ds = dataset

        


    for idx, example in enumerate(train_ds):
        processed = preprocess_with_filler_balanced(example, idx, filler_sentences, used_filler_sentences, tokenizer, quotes_list)
        if processed is None:
            break
        train_data.append(processed)



    train_data_path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\datasets\traintest.jsonl"
    with open(train_data_path, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


    print(f"Laget totalt: [{Count_double_quotes}] dobble quote eksempler.")
    print(f"Laget totalt: [{count_single_Quote_mix}] single quote/filler mix eksempler")
    print(f"Laget totalt:  [{count_only_filler}] Ingen sitater eksempler")
    print(f"Laget totalt: [{count_single_quote_natural}] single quote uten mix eksempler")

   # from trunacte import trunacte_dataset
   # trunacte_dataset(train_data_path)
    print("DONE!")

    log("done")

if __name__ == "__main__":
    main()
