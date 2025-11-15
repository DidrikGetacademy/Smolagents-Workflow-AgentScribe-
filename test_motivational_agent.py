#!/usr/bin/env python3
"""
Script to test motivational text identification using smolagents with different system prompts.
Tests 3 different text chunks to identify qualifying motivational texts.
"""

import os
import sys
import yaml
import torch
import gc
from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel, tool, FinalAnswerTool
from utility.log import log

load_dotenv()

# Get API key
OPENAI_APIKEY = os.getenv("OPENAI_APIKEY")
if not OPENAI_APIKEY:
    raise ValueError("OPENAI_APIKEY not found in environment variables")


TEST_CHUNKS = [ """
[5630.92s - 5635.40s] so when you took when you're talking about before the people the way people change around joe rogan or
[5635.40s - 5643.00s] you know you're talking to a hot girl you know i remember i remember when you know you're my career
[5643.56s - 5648.44s] exposes me to types of people i've not been exposed to before not the type of people that are walking
[5648.44s - 5658.28s] around the estates i was walking around and it's your intention that defines the way you're going to
[5658.28s - 5663.80s] treat them people because if you're talking to a hot woman your intention is to try and get her in bed
[5663.80s - 5668.36s] by the end of the night obviously you're going to be stuttering your words you're trying to negotiate
[5668.36s - 5675.72s] a pretty serious deal you know what i'm trying to say who's to say you're enough or she's interesting
[5675.72s - 5682.92s] or she's interested or you know if your intention is just the same as having a conversation with the
[5682.92s - 5689.64s] next person at the bar you know then all of a sudden it's a more natural interaction because at the
[5689.64s - 5695.16s] end of the day she's going to like you or not and you're you know you're saying she's a hot woman
[5695.16s - 5701.48s] you're going to like her or not but i think if you if you become a man in a certain position and and you
[5701.48s - 5709.73s] don't feel like you're you you you don't have options so to speak then you're going to look into
[5709.73s - 5716.61s] who she actually is because a lot of hot women aren't what they look like it looks like a great pretty on
[5716.61s - 5722.13s] the outside yeah it looks like a great person with we that can bring great things into a person's life
[5722.77s - 5731.01s] um and i think some of them may neglect um their other attributes you know and it's dangerous it's
[5731.01s - 5738.21s] dangerous for anybody who has an outsized capacity anywhere because yes yeah it paul graham's got this
[5738.21s - 5745.25s] wonderful quote where he says uh a lot of people look at uh those who are successful jerks and assume
[5745.25s - 5750.45s] the reason they're successful is because they're a jerk but that's not true the reason they're a jerk
[5750.45s - 5754.69s] the reason that they are a jerk is because their success allows them to get away with it yeah yeah
[5754.69s - 5760.29s] yeah and it's the same as not being a very nice person or somebody that's rich and an asshole or a
[5760.29s - 5768.29s] girl who's hot and has um no loyalty um uh isn't very kind their richness or their hotness is what
[5768.29s - 5772.45s] permits them to get away with it that's not why they have achieved the things they've achieved
[5772.45s - 5777.73s] well it's like it's like being a boxer that's got a big punch right very powerful and he leans
[5777.73s - 5785.49s] on that attribute and he wins the area title and if he's lucky he gets the european title but to be a
[5785.49s - 5791.65s] world champion and to compete at world level like you need to have rounded off your skill set it's as
[5791.65s - 5798.21s] simple as that so you know i meet a really pretty girl and she's relied on that up until this point
[5798.21s - 5803.49s] that i'm speaking to her it's just so uninteresting because it it smells of that that's what it is
[5804.13s - 5808.93s] you know where and like you say you become successful you get some money and or fame or
[5808.93s - 5815.49s] whatever it is if you've sort of leaned on that and just you're identified with that and that's who you
[5815.49s - 5820.29s] are now and you've not gone and studied and you've not tried to heal and you've not tried to it just
[5820.29s - 5826.37s] becomes a very unimpressive interaction you know i'm not interesting i'm not regulated i'm not kind i'm not
[5826.37s - 5835.01s] self-aware i'm not giving uh you mentioned that story of the guys who tried to rob your house oh
[5835.01s - 5852.02s] yeah we talk about that if you like i do want to yes what happened i mean look there was there were
[5852.02s - 5858.50s] some um like teenagers i've just moved into this house i made a big thing online i'm getting a new
[5858.50s - 5862.90s] house because this is the game right this is the rap game you're going to get to a place where you're
[5862.90s - 5868.98s] doing well and people want to know if it's true or not did you have you really transcended
[5869.86s - 5876.10s] the bottom is that really possible because that it's like in the batman film right and there's that Whats the best advice you ever
[5876.10s - 5881.22s] heard or recieved believe in yourself what is the worst advice you ever heard or received it can't be done
""",
# riktig svar:
# [5731.01s - 5738.21s] paul graham's got this [5738.21s - 5745.25s] wonderful quote where he says uh a lot of people look at uh those who are successful jerks and assume [5745.25s - 5750.45s] the reason they're successful is because they're a jerk but that's not true the reason they're a jerk [5750.45s - 5754.69s] the reason that they are a jerk is because their success allows them to get away with it yeah yeah [5754.69s - 5760.29s] yeah and it's the same as not being a very nice person or somebody that's rich and an asshole or a [5760.29s - 5768.29s] girl who's hot and has um no loyalty um uh isn't very kind their richness or their hotness is what [5768.29s - 5772.45s] permits them to get away with it that's not why they have achieved the things they've achieved [5772.45s - 5777.73s] well it's like it's like being a boxer that's got a big punch right very powerful and he leans [5777.73s - 5785.49s] on that attribute and he wins the area title and if he's lucky he gets the european title but to be a [5785.49s - 5791.65s] world champion and to compete at world level like you need to have rounded off your skill set it's as [5791.65s - 5798.21s] simple as that #
# [5869.86s - 5876.10s]  Whats the best advice you ever [5876.10s - 5881.22s] heard or recieved believe in yourself what is the worst advice you ever heard or received it can't be done


 """
[1961.36s - 1965.44s] and then didn't get to go to the product showcase last night for the same thing so yeah you're right
[1965.44s - 1975.04s] it's 95 98 of it is all of this stuff which is me looking at a laptop or me with a kindle or me with
[1975.04s - 1978.98s] the book or me with notes or me walking and going what the fuck do i think about that thing like
[1978.98s - 1983.56s] what's the idea what's the name for that concept what's the whatever and then okay i get to come
[1983.56s - 1990.00s] back and and finally have some fun but there's a a cool story from atomic habits by james clear
[1990.00s - 1996.74s] and uh i think it was the lead coach of maybe the chinese weightlifting team or maybe this is ben
[1996.74s - 2002.88s] bergeron's chasing excellence it's one of the two and they spoke to uh the lead coach of chinese
[2002.88s - 2006.90s] weightlifting team they said what's the difference between the guys who are elite and the guys that
[2006.90s - 2013.04s] are world champions and he said it's the world champions that are prepared to do the most boring
[2013.04s - 2020.84s] work with the least amount of complaining and uh i just really like that idea because especially as
[2020.84s - 2025.54s] things become more successful and this is like a a really unpopular talking point on the internet
[2025.54s - 2035.86s] because everybody wants to hear the zero to one not the 50 to 55 or 80 to 85 or 95 to 96 they don't
[2035.86s - 2039.82s] want to hear about the top end stuff because it doesn't sound it sounds more exclusionary right
[2039.82s - 2044.16s] because by definition most people are on the come up as opposed to closer to the top but i think it's
[2044.16s - 2048.52s] important to give people an idea of what are the pitfalls that are coming up for you if you do what
[2048.52s - 2052.92s] you say that you want to achieve which is become better in whatever it is that you even becoming
[2052.92s - 2058.18s] better as a parent becoming better as a dog owner becoming better as a friend or whatever as you
[2058.18s - 2062.36s] become more advanced there will be fewer and fewer people that can give you advice about what you're
[2062.36s - 2068.08s] going to come up against anyway point is people that are prepared to show up and do boring things
[2068.08s - 2076.92s] as their situation becomes more luxurious uh feels sort of opulent you know if you're a world champion
[2076.92s - 2082.16s] chinese weightlifter i imagine you've probably got your meals cooked for you you'll have body work
[2082.16s - 2085.62s] you'll have a coach you'll have a mindset person you'll have friends you'll be living in a house
[2085.62s - 2092.04s] that's probably quite nicely put together a bed that's constructed for your particular physiology
[2092.04s - 2100.64s] firmness all that sort of stuff and you think how do you remain hungry to go in and do an hour of
[2100.64s - 2109.42s] mobility work four times a week when you have your meals cooked for you and you've got a custom built
[2109.42s - 2116.34s] pillow and stuff and that uh preparedness to accept no matter how good i get at this thing
[2116.34s - 2124.12s] i will always have to do boring shit and that's not a bug it's a feature and not only is it a feature
[2124.12s - 2131.76s] it's a source of competitive advantage because as things get more salubrious as you as you raise up
[2131.76s - 2136.62s] through the ranks other people will have the same thought that you do which is i shouldn't need to do this
[2136.62s - 2141.14s] anymore you go okay so if you can continue to lean into the stuff that you did at the start
[2141.14s - 2147.02s] at the end that's where the competitive advantage lies yeah i was i was just doing the math to make
[2147.02s - 2152.84s] sure i knew what it all added up to but not even including because before the bodybuilding shows i just
[2152.84s - 2159.46s] did like i was dieting for like five months but just counting the year from january until that first
[2159.46s - 2171.04s] show a three month period there was five days straight of cardio it was a hundred it was 120
[2171.04s - 2178.32s] hours because it was um it was 30 minutes in the morning and 30 minutes after the workout for two of
[2178.32s - 2184.74s] those months and then for one of those months was an hour in the morning and an hour after so five like
[2184.74s - 2190.62s] who would even be able to watch five hours or something like that let alone or five days of
[2190.62s - 2196.14s] something like that let alone do it and that's the stuff i don't show off because i mean for one
[2196.14s - 2203.26s] nobody would even care to watch it but it's um it's like a it's like a hidden quest in a video game
""",
# riktig svar:
# # ingenting verdt og lagre


"""
[2388.08s - 2393.68s] where a lot of rappers or famous people fall to bits because they don't fill that middle bit with
[2393.68s - 2399.28s] anything productive they go to the clubs they see different women you know there's no structure to
[2399.28s - 2405.12s] it so they fall to bits so boxing has played sort of a role in just sort of keeping me focused until
[2405.12s - 2409.68s] sort of the next assignment anyway i get into this flipping big fuffle with this guy
[2410.96s - 2417.60s] and then i come out of the situation and because of the way that the situation went he wanted retribution
[2417.60s - 2425.20s] or whatever you'd call it and it was it was i took motivation from that the fact that this guy says when
[2425.20s - 2430.48s] when i see you i'm going to do whatever i would take motivation from situations like that you sort of
[2430.48s - 2437.76s] climb in your career and them situations subside and you're right and you're just dealing with
[2439.28s - 2448.72s] your own ambition your own sort of goals and you're you're no longer fueled by fear or you know anxiety
[2448.72s - 2456.32s] or whatever is chip on your shoulder yeah and and i did i found that difficult i and that's where the
[2456.32s - 2463.92s] whole sort of be inspired model sort of came from um because if i see people have done well for
[2463.92s - 2470.80s] themselves i want to i want to study the history how did he do well for himself why did he do well for
[2470.80s - 2478.56s] himself where did it go wrong did it go wrong you know and in in studying different people the roman
[2478.56s - 2486.16s] empire emperors kings you know i sort of seen the people that failed and the patterns as to why they failed
[2486.72s - 2493.28s] and the people that sustained and remained impressive throughout the life what are some of those
[2493.28s - 2502.00s] patterns that you've noticed vices vices and that's why i say and begin what i sort of seen is that some
[2502.00s - 2508.08s] of them people didn't come from the streets or the miners they had good parents and nice life but life
[2508.08s - 2513.28s] still presents bits of pain right and i think that undertone garba mate is quite good around this topic
[2513.28s - 2523.28s] i think the sort of undertone um to any vice is like a sort of low-level pain it's like having a
[2523.28s - 2527.20s] you know the only reason you are not living the life you want is because you day by day
[2527.20s - 2540.64s]  keep on feeding the life that you don't want to live um but then people do it by way of vices gambling drinking drugs smoking all of the vices um and i i feel like
[2540.64s - 2549.12s] the advice is grab people seven deadly sins lust gets people greed gets people and i think and i think
[2549.12s - 2554.00s] you you're attacked by them things i think the higher you sort of climbing towards doing something
[2554.88s - 2561.20s] positive for the world and sort of instilling love back into a world that's always like battling and
[2561.20s - 2568.24s] battling each other i think the more you're attacked by these um you know lust greed oh that's
[2568.24s - 2574.96s] interesting that's interesting that's interesting that it is a counterweight as you put things into
[2574.96s - 2581.68s] the world that you think are good you are tempted by ever more seductive and ever more advanced let's
[2581.68s - 2588.32s] give it biblical framework right which is the right place because we've got these these um stained glass
[2588.32s - 2596.16s] windows you know if you if you look at it from a position of if you're if you believe you're a
[2596.16s - 2601.60s] soldier of god you're doing god's work on the planet and that's just being a good person you know helping
[2601.60s - 2607.76s] contributing you might be a teacher you might do charity work you might do a podcast and inspire people
[2607.76s - 2613.68s] music whatever it is and your intention is to be as helpful to the planet as you possibly can
[2613.68s - 2621.52s] and we're using this biblical framework and seeing it like it's a film then you know it's as if it's as if
[2621.52s - 2628.64s] demons are now triggered by you and the light that you carry and they're like how can we stop him seeing
[2628.64s - 2635.68s] his highest potential because if he sees it he's going to create more light and the shadow a shadow
[2635.68s - 2643.28s] can't exist in the light you know so you're a threat so you get attacked from all different angles you
[2643.28s - 2649.52s] know and if we're talking quite spiritually there in the sort of physical world that comes by way of
[2649.52s - 2654.16s] lots more girls approaching out but it's not girls with good intentions lots more friends want to be
[2654.16s - 2658.08s] around you know but they ain't got good intentions either you're not like i say the cheesecake looks
[2658.08s - 2662.96s] a bit nicer the nightclub seems a bit more interesting you know and without saying any names we've we've
[2662.96s - 2669.44s] all seen seen people that have fell that's throughout history um and if you look at it it's the same thing
[2669.44s - 2675.44s] that over and over again that that grabs them and pulls them so on the flip side of that coin
"""
# riktig svar:
# [2523.28s - 2527.20s] the only reason you are not living the life you want is because you day by day[2527.20s - 2540.64s]  keep on feeding the life that you don't want to live
#
#

]

# Define correct answers for each chunk for scoring
# Each chunk can have multiple separate saves (list of texts) or None for no saves
CORRECT_ANSWERS = [
    # Chunk 1 correct answers (two separate motivational segments that should be saved separately)
    [
        """[5731.01s - 5738.21s] paul graham's got this [5738.21s - 5745.25s] wonderful quote where he says uh a lot of people look at uh those who are successful jerks and assume [5745.25s - 5750.45s] the reason they're successful is because they're a jerk but that's not true the reason they're a jerk [5750.45s - 5754.69s] the reason that they are a jerk is because their success allows them to get away with it yeah yeah [5754.69s - 5760.29s] yeah and it's the same as not being a very nice person or somebody that's rich and an asshole or a [5760.29s - 5768.29s] girl who's hot and has um no loyalty um uh isn't very kind their richness or their hotness is what [5768.29s - 5772.45s] permits them to get away with it that's not why they have achieved the things they've achieved [5772.45s - 5777.73s] well it's like it's like being a boxer that's got a big punch right very powerful and he leans [5777.73s - 5785.49s] on that attribute and he wins the area title and if he's lucky he gets the european title but to be a [5785.49s - 5791.65s] world champion and to compete at world level like you need to have rounded off your skill set it's as [5791.65s - 5798.21s] simple as that""",
        """[5869.86s - 5876.10s] Whats the best advice you ever [5876.10s - 5881.22s] heard or recieved believe in yourself what is the worst advice you ever heard or received it can't be done"""
    ],

    # Chunk 2 correct answer (should be nothing - no motivational content)
    None,  # "ingenting verdt og lagre" = nothing worth saving

    # Chunk 3 correct answer (single motivational segment)
    [
        """[2523.28s - 2527.20s] the only reason you are not living the life you want is because you day by day[2527.20s - 2540.64s]  keep on feeding the life that you don't want to live"""
    ]
]

# Global scoring variables - initialized at module level
current_chunk_index = 0
total_score = 0
max_possible_score = 0
saved_texts_for_current_chunk = []  # Track all saves for current chunk
chunk_completed = False  # Flag to prevent double counting

def calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts based on key timestamps and content."""
    if text1 is None or text2 is None:
        return 1.0 if text1 == text2 else 0.0

    # Extract timestamps from both texts
    import re
    timestamps1 = re.findall(r'\[[\d\.]+s - [\d\.]+s\]', text1)
    timestamps2 = re.findall(r'\[[\d\.]+s - [\d\.]+s\]', text2)

    # Calculate overlap in timestamps
    timestamp_overlap = len(set(timestamps1) & set(timestamps2)) / max(len(set(timestamps1) | set(timestamps2)), 1)

    # Calculate content similarity (simple word overlap)
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    content_overlap = len(words1 & words2) / max(len(words1 | words2), 1)

    # Combined similarity score
    return (timestamp_overlap * 0.7 + content_overlap * 0.3)

def find_best_match_in_correct_answers(saved_text: str, correct_answers_for_chunk) -> tuple:
    """Find the best matching correct answer for a saved text."""
    if correct_answers_for_chunk is None:
        return False, 0.0, "No correct answers for this chunk"

    if not isinstance(correct_answers_for_chunk, list):
        # Single correct answer
        similarity = calculate_text_similarity(saved_text, correct_answers_for_chunk)
        return similarity >= 0.8, similarity, "Single answer"

    # Multiple correct answers - find best match
    best_similarity = 0.0
    best_match_index = -1

    for i, correct_text in enumerate(correct_answers_for_chunk):
        similarity = calculate_text_similarity(saved_text, correct_text)
        if similarity > best_similarity:
            best_similarity = similarity
            best_match_index = i

    is_match = best_similarity >= 0.8
    match_info = f"Best match: Answer {best_match_index + 1}" if best_match_index >= 0 else "No good match"

    return is_match, best_similarity, match_info

@tool
def SaveMotivationalText(text: str, text_file: str) -> str:
    """A function/tool that saves qualifying motivational texts for a punchy 15-30 second shorts video.
    Args:
        text (str): The complete motivational text block to save.
            - The text must evoke emotional energy (e.g., hope, empathy, vulnerability) alongside inspiration, making it suitable for grabbing a listener's attention in a short video.
            - The text must be a self-contained, inspirational passage that encourages action, resilience, discipline, perseverance, or personal growth, suitable for a short video without relying on prior context or external references.
            - Every line must include the exact timestamp range from the original chunk (e.g., '[start1 - end1] Text1 [start2 - end2] Text2').
            - You must provide the text exactly as it appears in the chunk, preserving every word, space, and punctuation, with no rephrasing, omissions, or alterations, except that leading conjunctions or transitional words (e.g., 'And,' 'So,' 'But') may be removed if doing so makes the text fully self-contained without changing the core motivational message.
            - If the saved text begins or ends in the middle of a line, the timestamp from that line must still be included.
            - The number of lines is not fixed; include all lines (with timestamps) that the text spans, whether it is one line or many.
            - Wrap the entire block in triple quotes if it contains commas, quotes, or line breaks to ensure proper handling in Python.
        text_file (str): The path to the text file where the motivational text will be saved.

    Returns:
        str: Confirmation message with scoring information
    """
    global saved_texts_for_current_chunk

    # Add this text to the current chunk's saved texts
    saved_texts_for_current_chunk.append(text.strip())

    # Get correct answers for current chunk
    correct_answers_for_chunk = CORRECT_ANSWERS[current_chunk_index] if current_chunk_index < len(CORRECT_ANSWERS) else None

    # Find best match for this saved text
    is_correct, similarity_score, match_info = find_best_match_in_correct_answers(text.strip(), correct_answers_for_chunk)

    # Prepare scoring information
    score_info = f"""
SAVE #{len(saved_texts_for_current_chunk)} FOR CHUNK {current_chunk_index + 1}:
- Status: {'✅ CORRECT MATCH' if is_correct else '❌ NO MATCH'}
- Similarity Score: {similarity_score:.2%}
- Match Info: {match_info}
- Saved Texts So Far: {len(saved_texts_for_current_chunk)}
"""

    # Save to file with scoring information
    with open(text_file, "a", encoding="utf-8") as f:
        f.write(f"===START_TEXT_{len(saved_texts_for_current_chunk)}===\n")
        f.write(text.strip())
        f.write("\n===END_TEXT===\n")
        f.write(score_info)
        f.write("="*50 + "\n\n")

    log(f"Saved text #{len(saved_texts_for_current_chunk)}: {text[:100]}...")
    log(f"Score info: {score_info}")

    return f"✅ Motivational text #{len(saved_texts_for_current_chunk)} saved! {match_info} (Similarity: {similarity_score:.1%})"

@tool
def RejectChunk(reason: str, text_file: str) -> str:
    """
    Tool to use when no qualifying motivational text is found in a chunk.

    Args:
        reason (str): Explanation of why the chunk was rejected
        text_file (str): The path to the text file for logging

    Returns:
        str: Confirmation message with scoring information
    """
    global saved_texts_for_current_chunk

    # Check if rejection is correct for this chunk
    correct_answers_for_chunk = CORRECT_ANSWERS[current_chunk_index] if current_chunk_index < len(CORRECT_ANSWERS) else None
    is_correct_rejection = correct_answers_for_chunk is None

    # Prepare scoring information
    score_info = f"""
CHUNK REJECTION:
- Chunk {current_chunk_index + 1}: {'✅ CORRECT REJECTION' if is_correct_rejection else '❌ INCORRECT REJECTION'}
- Reason: {reason}
- Expected: {'No saves (correct)' if correct_answers_for_chunk is None else f'{len(correct_answers_for_chunk)} saves expected'}
"""    # Log the rejection
    with open(text_file, "a", encoding="utf-8") as f:
        f.write("===CHUNK_REJECTED===\n")
        f.write(f"Reason: {reason}\n")
        f.write("===END_REJECTION===\n")
        f.write(score_info)
        f.write("="*50 + "\n\n")

    log(f"Chunk rejected: {reason}")
    log(f"Score info: {score_info}")

    return f"❌ Chunk rejected: {reason} | {'✅ Correct rejection' if is_correct_rejection else '❌ Should have saved content'}"

# Different system prompts to test
with open(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\testing\structured_output_prompt_TranscriptReasoning_gpt5.yaml", "r", encoding="utf-8") as f:
    SYSTEM_PROMPTS = yaml.safe_load(f)








def calculate_chunk_score(saved_texts: list, correct_answers_for_chunk) -> tuple:
    """Calculate score for a completed chunk."""
    if correct_answers_for_chunk is None:
        # Should have no saves
        return (1, 1) if len(saved_texts) == 0 else (0, 1)

    if not isinstance(correct_answers_for_chunk, list):
        correct_answers_for_chunk = [correct_answers_for_chunk]

    # For each correct answer, check if we have a matching saved text
    matches = 0
    for correct_text in correct_answers_for_chunk:
        best_match = 0.0
        for saved_text in saved_texts:
            similarity = calculate_text_similarity(saved_text, correct_text)
            best_match = max(best_match, similarity)

        if best_match >= 0.8:
            matches += 1

    # Score is matches found / total expected
    total_expected = len(correct_answers_for_chunk)
    return (matches, total_expected)

def print_final_score_report(chunk_scores: list):
    """Print a comprehensive score report at the end."""
    total_score = sum(score for score, _ in chunk_scores)
    max_possible_score = sum(max_score for _, max_score in chunk_scores)

    percentage = (total_score / max_possible_score * 100) if max_possible_score > 0 else 0

    print("\n" + "="*60)
    print("🎯 FINAL SCORE REPORT")
    print("="*60)
    print(f"📊 Total Score: {total_score}/{max_possible_score} ({percentage:.1f}%)")
    print(f"📈 Performance Level: ", end="")

    if percentage >= 90:
        print("🏆 EXCELLENT")
    elif percentage >= 70:
        print("✅ GOOD")
    elif percentage >= 50:
        print("⚠️ NEEDS IMPROVEMENT")
    else:
        print("❌ POOR")

    print("\n📋 Chunk-by-Chunk Breakdown:")
    for i, (score, max_score) in enumerate(chunk_scores):
        correct_answers = CORRECT_ANSWERS[i]
        if correct_answers is None:
            expected = "NO SAVES (non-motivational)"
        elif isinstance(correct_answers, list):
            expected = f"{len(correct_answers)} SEPARATE SAVES"
        else:
            expected = "1 SAVE"

        print(f"  Chunk {i+1}: {score}/{max_score} - Expected: {expected}")

    print(f"\n📄 Detailed results saved to: testing/test.txt")
    print("="*60)

if __name__ == "__main__":
    # Reset global variables
    current_chunk_index = 0
    total_score = 0
    max_possible_score = 0
    saved_texts_for_current_chunk = []
    chunk_completed = False

    # Clear the output file at start
    test_file = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\testing\test.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(f"MOTIVATIONAL TEXT IDENTIFICATION TEST RESULTS\n")
        f.write(f"Test Date: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}\n")
        f.write("="*60 + "\n\n")

    model = LiteLLMModel(
        model_id="gpt-5",
        api_key=OPENAI_APIKEY,
        max_tokens=16000,
        reasoning_effort="high",
    )

    # Create the agent with the current prompt
    agent = CodeAgent(
        model=model,
        tools=[SaveMotivationalText, RejectChunk, FinalAnswerTool()],
        max_steps=1,
        prompt_templates=SYSTEM_PROMPTS,
        verbosity_level=4,
        use_structured_outputs_internally=True
    )

    try:
        print("🚀 Starting Motivational Text Identification Test")
        print(f"📝 Testing {len(TEST_CHUNKS)} chunks with scoring enabled")

        chunk_scores = []

        for i, chunk in enumerate(TEST_CHUNKS):
            current_chunk_index = i  # Update global chunk index
            saved_texts_for_current_chunk = []  # Reset for each chunk

            print(f"\n🔍 Processing Chunk {i+1}/{len(TEST_CHUNKS)}")

            # Show expected results for this chunk
            expected = CORRECT_ANSWERS[i]
            if expected is None:
                print(f"📋 Expected: NO SAVES (non-motivational content)")
            elif isinstance(expected, list):
                print(f"📋 Expected: {len(expected)} SEPARATE SAVES")
            else:
                print(f"📋 Expected: 1 SAVE")

            task = f"""Your task is to Identify Qualifying Motivational Texts & Save them if any is found in the chunk.
                        Here is the chunk you must analyze:
                        [chunk start]
                        {chunk}
                        [chunk end]"""

            response = agent.run(task=task, additional_args={"text_file": test_file})

            # Calculate score for this chunk
            chunk_score = calculate_chunk_score(saved_texts_for_current_chunk, CORRECT_ANSWERS[i])
            chunk_scores.append(chunk_score)

            print(f"✅ Chunk {i+1} completed: {chunk_score[0]}/{chunk_score[1]} correct")

        # Print final results
        print_final_score_report(chunk_scores)

    except Exception as e:
        print(f"❌ Error during execution: {str(e)}")

    finally:
        gc.collect()
        torch.cuda.empty_cache()
        del agent
        del model
