This project idea and workflow is inspired by one simple goal: saving time. Instead of manually downloading a 1-hour and 52-minute podcast video and searching through it to find moments where wisdom, motivational quotes, or inspiring messages are spoken—just to clip 10–15 second segments for short videos—I’ve created an automated multi-agent workflow using the smolagents library.

The system uses a manager agent and a reasoning agent, along with custom tools that:

Extract audio from video files

Transcribe the audio into text using Whisper

Save the transcript to a .txt file

The manager agent then feeds chunks of the transcript to the reasoning agent, which analyzes the content and identifies potential moments of wisdom, quotes, or inspirational messages—saving them with timestamps—until the entire transcript has been processed.

This repo also includes logic for fine-tuning the reasoning process, helping the agent better understand which segments to keep and which to skip.