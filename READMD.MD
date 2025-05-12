This project idea and workflow is inspired by one simple goal: saving time. Instead of manually downloading a 1-hour and 52-minute podcast video and searching through it to find moments where wisdom, motivational quotes, or inspiring messages are spoken—just to clip 10–15 second segments for short videos—I’ve created an automated multi-agent workflow using the smolagents library.

The system uses a manager agent and a reasoning agent, along with custom tools that:

Extract audio from video files

Transcribe the audio into text using Whisper - onnx format 8 int 16 model 

Save the transcript to a .txt file

The manager agent then feeds chunks of the transcript to the reasoning agent, which analyzes the content and identifies potential moments of wisdom, quotes, or inspirational messages—saving them with timestamps—until the entire transcript has been processed. This saves alot of time. now i can just look at the saved file where the reasoning agent saved text with timestamps and just copy and paste them into my short video maker. This is a huge time saver

This repo also includes logic for fine-tuning the reasoning process, helping the agent better understand which segments to keep and which to skip.



After installing smolagents libary with 
pip install smolagents
pip install smolagents[transformers]
pip install smolagents[whisper]

you have too change current class/add class in smolagents.model, check out Smolagents_libary_Changes.txt file.
changes that has been made in smolagents.model is workarounds too make it work with loading local models, loading in c2translate models and  speectotexttool in smolagents.tools for the whisper model.
soon i will add logic for loading onnx models with genai  in smolagents.model but you know its a reason smolagents have not added class sutch model loading, but i will figure it out, just need some ibux and time
