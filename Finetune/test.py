import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=r"C:\Users\didri\Desktop\LLM-models\LLM-Models\mistralai\Mistral-7B-Instruct-v0.3",
    trust_remote_code=True
)

file_path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Finetune\Dataset_detecting_motivationalquotes_from_chunk\Supervised_dataset\train.jsonl"

with open(file_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)  # loads entire JSON array at once

for i, row in enumerate(dataset):
    if i >= 2:  # only first 2 examples
        break
    messages = [
        {"role": "system", "content": row["system_prompt"]},
        {"role": "user", "content": row["instruction"] + row["input_text"]},
        {"role": "assistant", "content": row["label_text"]}
    ]

    # Add user input(s)



    prompt_str = tokenizer.apply_chat_template(conversation=messages, tools=None, tokenize=False)
    print(f"=== Prompt {i+1} ===")
    print(prompt_str)
    print("\n-----\n")
# === Prompt 1 ===
# <s>[INST] 
#     Save standalone motivational quotes.
#     Analyze the chunk between [chunk start] and [chunk end].
#     If you find one or more qualifying quotes, provide a Thought explaining your analysis, save each quote separately with SaveMotivationalText including all timestamps for that quote, and finish with final_answer().
#     If no quotes are found, provide a Thought stating no suitable quotes were found, and just use final_answer().
#     [600.00s - 603.40s] And, you wanna set up the segue, of where we can go to this next part?.
# [603.50s - 607.08s] Yeah, but you remember, like, the mid-1960s, Simula.
# [607.18s - 610.43s] And we got to talking about the convict list system.
# [610.53s - 613.97s] My friend, you know, Forbes or whatever, has, like, 15 other friends coming in that, that are trying to sell me things before I can even talk to them.
# [614.07s - 617.23s] give children the freedom to define their own niches and imagine doing things that, that don't exist or aren't currently considered possible.
# [617.33s - 620.17s] One of the, the, in, in stages of research, right? You have to come up with a great research question.
# [620.27s - 623.26s] And I'm not a big tomato fan, raw.
# [623.36s - 625.91s] Mmm, bi-weekly, well, let's see.
# [626.01s - 628.76s] And, you wanna set up the segue, of where we can go to this next part?.
# [628.86s - 631.71s] Yeah, but you remember, like, the mid-1960s, Simula.
# [631.81s - 635.18s] And we got to talking about the convict list system.
# [635.28s - 638.24s] My friend, you know, Forbes or whatever, has, like, 15 other friends coming in that, that are trying to sell me things before I can even talk to them.
# [638.34s - 641.67s] give children the freedom to define their own niches and imagine doing things that, that don't exist or aren't currently considered possible.
# [641.77s - 644.47s] One of the, the, in, in stages of research, right? You have to come up with a great research question.
# [644.57s - 648.09s] And I'm not a big tomato fan, raw.
# [648.19s - 650.67s] Mmm, bi-weekly, well, let's see.
# [/INST] Thought: No Standalone Quote found in the chunk. so i will not save anything and just provide a final answer
# <code>
# final_answer("After carefully analysing the chunk/text, i have concluded nothing can be saved.")
# </code></s>

# -----

# === Prompt 2 ===
# <s>[INST]
#     Save standalone motivational quotes.
#     Analyze the chunk between [chunk start] and [chunk end].
#     If you find one or more qualifying quotes, provide a Thought explaining your analysis, save each quote separately with SaveMotivationalText including all timestamps for that quote, and finish with final_answer().
#     If no quotes are found, provide a Thought stating no suitable quotes were found, and just use final_answer().
#     [chunk start]
# [600.00s - 602.54s] But projected on a quite a sort of creased black fabric background as well, which is the polar opposite to the white that we would normally expect.
# [602.64s - 605.75s] And that has, uh, also, sort of,.
# [605.85s - 609.22s] We had 15 in the current cohort.
# [609.32s - 612.64s] Window was open, you know, the sun was setting or something, and I just remember having this eureka moment of being like,.
# [612.74s - 615.95s] Um, you asked, or you mentioned,.
# [616.05s - 618.76s] And that's, I'm like, hey, I'm, I'm grateful for it.
# [618.86s - 621.90s] Don't let that stop you from moving ahead and growing your business right now.
# [622.00s - 625.00s] Precious moments with precious people can help you realize how precious life can be
# [625.10s - 627.90s] But projected on a quite a sort of creased black fabric background as well, which is the polar opposite to the white that we would normally expect.
# [628.00s - 631.45s] And that has, uh, also, sort of,.
# [631.55s - 634.11s] We had 15 in the current cohort.
# [634.21s - 637.27s] Window was open, you know, the sun was setting or something, and I just remember having this eureka moment of being like,.
# [637.37s - 639.82s] Um, you asked, or you mentioned,.
# [639.92s - 642.95s] And that's, I'm like, hey, I'm, I'm grateful for it.
# [643.05s - 646.24s] Don't let that stop you from moving ahead and growing your business right now.
# [chunk end]
# [/INST] Thought: I found 1 standalone motivational passagethat meet the criteria, so I’m saving it.
# <code>
# SaveMotivationalText(text="[622.00s - 625.00s] Precious moments with precious people can help you realize how precious life can be",text_file=text_file)
# final_answer("im done analyzing chunk")
# </code></s>

# -----

# PS C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing> 