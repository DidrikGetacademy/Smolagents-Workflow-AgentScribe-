import json
import re
# ✅ Correctly separated fields:
texts_to_insert = [
    {
        "tekst":
        """
        [3.31s - 10.75s] Today, we've got a crazy cool topic. We are talking about mindset. Your mind is either
        [10.75s - 16.15s] working for you or against you. That's what it's doing. So whether you're listening to this episode
        [16.15s - 21.83s] because you struggle right now with overthinking or feeling unworthy, or maybe you have a really
        [21.83s - 27.51s] positive outlook, but you just want to level up. You want to play a bigger game. That's where I am
        [27.51s - 33.71s] right now. So today you and I are going to get serious about making your mind work for you.
        [35.79s - 44.45s] Hey, it's your friend Mel and welcome to a mind bending and really cool episode of the Mel
        [44.45s - 53.07s] Robbins podcast. Okay. I wanted to just start today by saying thank you. Thank you. Thank you. Thank you
        [53.07s - 61.29s] to you. I often say that the Mel Robbins podcast is not my podcast. It's our podcast because this is a
        [61.29s - 65.17s] conversation between you and me. And I wanted to start off by saying thank you because
        [65.17s - 76.25s] about 90 seconds ago, I got word that you have voted the Mel Robbins podcast as the most inspirational
        [76.25s - 85.07s] podcast of 2022. And we have won the listener's choice award for the most inspiring podcast of 2022.
        [85.07s - 92.63s] That is a huge deal because we just launched two and a half months ago. So from the bottom of my heart
        [92.63s - 98.71s] on behalf of my team, I just wanted to thank you. I wanted to thank you for showing up, for listening,
        [98.95s - 104.67s] for sharing these episodes with friends and family members, for giving us feedback, for asking questions,
        [104.67s - 112.57s] for submitting topic ideas. This podcast is changing people's lives and it is inspiring and empowering
        [112.57s - 118.55s] people around the world because of you. So thank you. And if you're brand new to the Mel Robbins podcast
        [118.55s - 125.61s] and this amazing, energizing group of people that listen to this podcast, I want to say welcome.
        [126.07s - 131.50s] I'm Mel Robbins. I'm a New York Times bestselling author. And I'm one of the most trusted experts in the
        [131.50s - 138.72s] world on behavior change and motivation. And today we've got a crazy cool topic. We are talking about
        [138.72s - 144.18s] mindset. And before we jump into the science and the cool tactics that you're going to be able to apply
        [144.18s - 150.40s] to your life to change your mindset, I want to just remind you that this episode is part of a month
        [150.40s - 156.12s] long series that we are doing here on the Mel Robbins podcast about the building blocks and the research
        [156.12s - 162.36s] that you need to know in order to create a better life. Here's the simple truth about your mindset.
        [163.14s - 171.28s] Your mind is either working for you or against you. That's what it's doing. And so by the end of
        [171.28s - 176.36s] today's episode, there's going to be a couple things that go down. First of all, you are going to
        [176.36s - 183.80s] understand that you have the power to reprogram your mind. That's right. You can take simple steps
        [183.80s - 190.94s] and you can practice them every day to train your mind to work for you. And I'm also going to prove
        [190.94s - 195.92s] to you today using very simple science that your mind is trying to help you. It doesn't know any
        [195.92s - 202.18s] better if it's working against you. And when you can identify the way that you want to feel or what
        [202.18s - 208.32s] you want to do with your life, you can change your mindset to help you. And when you do that, here's
        [208.32s - 214.42s] what's super cool. It improves the day-to-day experience of your life and it changes what it's
        [214.42s - 220.16s] like to be in your head. So whether you're listening to this episode because you struggle right now with
        [220.16s - 226.80s] overthinking or feeling unworthy, or maybe you have a really positive outlook, but you just want to level
        [226.80s - 233.60s] up. You want to play a bigger game. That's where I am right now. I am so ready to take a bigger swing
        [233.60s - 239.86s] to knock it out of the park this year. And the mindset and creating a more powerful mindset,
        [239.86s - 247.62s] that is a tool in your arsenal to help you achieve anything that you want. So today, you and I are
        [247.62s - 254.20s] going to get serious about making your mind work for you. And I want to start us off with a question
        [254.20s - 261.06s] from a listener named Brandy. Hi there, Mel. My name is Brandy. How do I stop the spiral of negative
        [261.06s - 266.62s] thoughts and feelings? I really want to reset and start embracing a happier life. I just don't know
        [266.62s - 271.68s] where to start. I hope you can help. Brandy, I am so happy that you asked this question because
        [271.68s - 278.54s] we have received thousands of versions of this question. And I'm also kind of thrilled. I picked
        [278.54s - 284.72s] your question in particular because you use the word reset. And today, I am going to teach you how to
        """,

        "quote": "[156.12s - 162.36s] Here's the simple truth about your mindset. [163.14s - 171.28s] Your mind is either working for you or against you.",

        "Thought": "The overall context of the chunk is a motivational podcast episode discussing the importance of mindset and how to control it. The speaker, Mel Robbins, is addressing a listener's question about how to stop negative thoughts and feelings. The text I am looking for to save should be a powerful, standalone quote or message that encapsulates the key insight or advice given in the episode. This text should be memorable and impactful, suitable for a motivational short video."

    }
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
