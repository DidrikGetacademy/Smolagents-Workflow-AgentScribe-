from smolagents import  CodeAgent,TransformersModel,FinalAnswerTool
import torch
import yaml
####################################################################################################################################################################################################################################
chunk_test1_should_extract_quotes = """
[2210.45s - 2215.81s] I had to run errands all day and forgot my umbrella, so I got soaked in the rain.  
[2215.82s - 2220.13s] Sometimes you just have to accept the weather and keep moving forward.  
[2220.14s - 2224.00s] People say, “The comeback is stronger than the setback.”  
[2224.01s - 2227.44s] But honestly, every step you take matters, no matter how small.  
[2227.45s - 2231.10s] I believe growth is quiet—like this quote from my mentor:  
[2231.11s - 2236.50s] “Success is built on a thousand quiet efforts, day after day,  
[2236.51s - 2240.00s] unseen by the world but felt deeply in the heart.”  
[2240.01s - 2243.30s] It’s not always obvious, but persistence pays off in the long run.  
[2243.31s - 2246.20s] Just yesterday, I bumped into an old friend and we talked about how life moves so fast.  
[2246.21s - 2249.50s] I guess sometimes you have to breathe and trust things will work out.  
[2249.51s - 2252.00s] But other times, it’s okay not to know exactly where you’re headed.  
[2252.01s - 2255.10s] After all, “Heaven is a place on earth, but you have to go through the storm to appreciate the sunshine.”  
[2255.11s - 2259.00s] That one always stuck with me, especially during tough times.  
[2259.01s - 2263.20s] Here’s something else that really resonates:  
[2263.21s - 2267.00s] “When the night is darkest,  
[2267.01s - 2270.80s] remember that the stars  
[2270.81s - 2274.50s] are shining just out of sight.”  
[2274.51s - 2278.00s] It’s a reminder that hope can exist even when things seem bleak.  
[2278.01s - 2282.10s] Life’s full of ordinary moments and sometimes extraordinary wisdom in between.  
[2282.11s - 2285.30s] You just have to listen closely to catch it.
"""


#CORRECT BEHAVIOUR: 
quotes = [
    "[2220.14s - 2224.00s] People say, “The comeback is stronger than the setback.”",
    "[2231.11s - 2240.00s] “Success is built on a thousand quiet efforts, day after day, unseen by the world but felt deeply in the heart.”",
    "[2252.01s - 2255.10s] “After all, Heaven is a place on earth, but you have to go through the storm to appreciate the sunshine.”",
    "[2263.21s - 2274.50s] “When the night is darkest,\nremember that the stars\nare shining just out of sight.”"
]


#Model run with no system prompt modification & deep detailed task instruction
Model_result = """
  quotes = [                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
      "[2215.82s - 2220.13s] Sometimes you just have to accept the weather and keep moving forward.",                                                                                                                                                                                                                                                                                                                                                                                          
      "[2231.11s - 2236.50s] Success is built on a thousand quiet efforts, day after day, unseen by the world but felt deeply in the heart.",                                                                                                                                                                                                                                                                                                                                                  
      "[2255.11s - 2259.00s] That one always stuck with me, especially during tough times.",                                                                                                                                                                                                                                                                                                                                                                                                   
      "[2267.01s - 2270.80] Remember that the stars are shining just out of sight.",                                                                                                                                                                                                                                                                                                                                                                                                           
      "[2282.11s - 2285.30] You just have to listen closely to catch it."                                                                                                                                                                                                                                                                                                                                                                                                                      
  ]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
""" ##SUMMARY ---> BAD REASONING AND EXTRACTION...



#Model run with system prompt modification
Model_result = """ 



"""




####################################################################################################################################################################################################################################
chunk_test2_should_extract_quotes = """
[3000.00s - 3004.50s] Life has its ups and downs, but it’s how you respond that defines you.  
[3004.51s - 3008.75s] I remember hearing, “Your mindset shapes your reality.”  
[3008.76s - 3012.00s] Sometimes, the smallest step in the right direction ends up being the biggest step of your life.  
[3012.01s - 3015.20s] You might not see progress every day, but persistence wins in the long run.  
[3015.21s - 3018.55s] Honestly, some days I just feel stuck, and that’s part of the journey.  
[3018.56s - 3022.10s] A wise friend once said, “Doubt kills more dreams than failure ever will.”  
[3022.11s - 3025.80s] Not every piece of advice fits everyone — you have to find what works for you.  
[3025.81s - 3029.00s] The real strength is in getting back up after you fall.  
[3029.01s - 3032.50s] I guess, sometimes you just need to listen to your own voice instead of others’.  
[3032.51s - 3036.00s] Remember, “Success is not final, failure is not fatal: It is the courage to continue that counts.”  
[3036.01s - 3039.45s] Life isn’t a race; it’s a marathon with its own pace.  
"""


#CORRECT BEHAVIOUR: 
quotes_Correct_behaviour_test2 = [
    "[3004.51s - 3008.75s] “Your mindset shapes your reality.”",
    "[3008.76s - 3012.00s] Sometimes, the smallest step in the right direction ends up being the biggest step of your life.",
    "[3012.01s - 3015.20s] You might not see progress every day, but persistence wins in the long run.",
    "[3018.56s - 3022.10s] “Doubt kills more dreams than failure ever will.”",
    "[3025.81s - 3029.00s] The real strength is in getting back up after you fall.",
    "[3032.51s - 3036.00s] “Success is not final, failure is not fatal: It is the courage to continue that counts.”",
    "[3036.01s - 3039.45s] Life isn’t a race; it’s a marathon with its own pace."
]





####################################################################################################################################################################################################################################
chunk_test3_no_potensial_quotes = """
[4000.00s - 4003.50s] Sometimes things just happen, and there’s not much to say about it.  
[4003.51s - 4006.75s] You know, it’s like when you try and don’t quite get it right.  
[4006.76s - 4010.00s] Maybe one day it will make sense, but not today.  
[4010.01s - 4013.20s] People say a lot of things, but it doesn’t always mean much.  
[4013.21s - 4016.55s] There’s always tomorrow, or so they say.  
[4016.56s - 4020.10s] Sometimes you just have to wait and see what happens next.  
[4020.11s - 4023.80s] It’s not really about winning or losing, it’s about the experience — or something like that.  
[4023.81s - 4027.00s] I guess every day is a new chance, but who’s really keeping track?  
"""

###Correct behaviour
print("You are done reviewing the chunk, and nothing is worth saving.")




####################################################################################################################################################################################################################################

chunk_test4_No_quote_extraction = """
[6000.00s - 6004.00s] So yesterday I went to the grocery store and forgot my shopping list, which made me wander around the aisles longer than I planned.  
[6004.01s - 6008.50s] The cashier was friendly though, which helped make the wait a little more pleasant than usual.  
[6008.51s - 6013.00s] After that, I took a slow walk around the park nearby, just enjoying the fresh air and watching the leaves fall.  
[6013.01s - 6017.20s] The weather was pretty mild—not too hot, not too cold—just a bit cloudy but nothing too gloomy.  
[6017.21s - 6022.00s] I bumped into an old friend from college while sitting on a bench, and we spent a few minutes catching up on life.  
[6022.01s - 6026.50s] Then I decided to grab a coffee from the café down the street before heading back home to relax.  
[6026.51s - 6030.00s] I spent the afternoon watching a couple of episodes from a series I’ve been slowly working my way through.  
[6030.01s - 6035.00s] Dinner was simple—just some pasta with tomato sauce and a fresh salad on the side, nothing fancy but satisfying enough.  
[6035.01s - 6039.50s] After eating, I cleaned up the kitchen and put the dishes away before settling down with a book for a bit.  
[6039.51s - 6044.00s] It was a quiet evening, nothing eventful happened, just the usual routine to wind down.  
[6044.01s - 6048.30s] Before going to bed, I spent some time planning out what I need to do for the next day.  
[6048.31s - 6052.00s] Nothing too exciting, just one of those ordinary days that pass by without much fanfare.
"""


###Correct behaviour
print("You are done reviewing the chunk, and nothing is worth saving.")

######################################################################################################################################################################################################################################################################################################################################################




Test_task = f"""
You are given a text chunk to carefully analyze for motivational or inspirational quotes, advice, or any meaningful content that conveys wisdom.  
Your goal is to identify text that could be used in a motivational short.  

If you find any such text,remember to include the timestamp & print it using the format:  
saving quote: "<quote>"  

If no quote is worth saving, print:  
You are done reviewing the chunk, and nothing is worth saving.  

Please analyze the chunk as a human reader would.  

The text chunk to analyze is:  
{chunk_test1_should_extract_quotes}
"""


Model = r"C:\Users\didri\Desktop\LLM-models\Qwen\Qwen_3B"
def test_model_reasoning_ability_on_Chunk(Model):
    Transformer_model = TransformersModel(
        model_id=Model,
        device_map="cuda",
        torch_dtype=torch.float16,
        load_in_8bit=True,
        trust_remote_code=True
    )

    with open(r"C:\Users\didri\Desktop\Programmering\Full-Agent-Flow_VideoEditing\Prompt_templates\Test_prompt_template.yaml",  "r", encoding="utf-8") as file:
        Agent_prompt_template = yaml.safe_load(file)


    agent = CodeAgent(
        model=Transformer_model,
        max_steps=10,
        tools=[FinalAnswerTool()],
        add_base_tools=True,
        prompt_templates=Agent_prompt_template,
        verbosity_level=4,
    )
    agent.run(task=Test_task)
    agent.visualize()
  

if __name__ == "__main__":
     test_model_reasoning_ability_on_Chunk(Model)



