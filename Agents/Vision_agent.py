
from Agents.utility.Agent_tools import extract_window_frames, get_recent_background_music_summary
import os
import json
from openai import OpenAI
from utility.log import log
from dotenv import load_dotenv
load_dotenv()
openai = OpenAI(api_key=os.getenv("OPENAI_APIKEY"))
def run_multi_Vision_model_agent(video_path , Additional_content, already_uploaded_videos,start_time,end_time) -> str:
    """Agent that selects background music for the motivational shorts video
    Generates a detailed motivational summary of a podcast using both text and audiovisual inputs.
    The function leverages a Vision-Audio model to process the video and audio content,
    and a text summarization pipeline to produce a concise, motivational summary.

    Args:
        _text: The transcript or textual content of the podcast to summarize.
        video_path: Path to the video file that is being uploaded of the podcast, used to extract visual frames for context.
        audio_path: Path to the audio file that is being uploaded of the podcast, used for extracting audio features for context.

    Returns:
        A string containing a detailed summary of the podcast, highlighting motivational points, emotions,
        and actionable advice. If summarization fails, a message indicating the failure is returned.
    """
    from PIL import Image
    frames = extract_window_frames(video_path, start_time=start_time, end_time=end_time, max_frames=5)


    audio_length = end_time - start_time
    if os.path.exists(already_uploaded_videos):
        try:
            with open(already_uploaded_videos, "r", encoding="utf-8") as f:
                json.load(f)  # validate file is readable json
            uploaded_videos_content = get_recent_background_music_summary(already_uploaded_videos, limit=3)
        except Exception:
            uploaded_videos_content = "No previous background tracks recorded."
    else:
        uploaded_videos_content = "No previous background tracks recorded."


    System_prompt = f"""
        You are specialized in music, speech, visual and textual reasoning with a focus on optimizing motivational podcasts. Given a list of candidate background music tracks,
        your task is to select the single most suitable instrumental music to be used as background music to enhance the Motivational shorts video's effect.
        YOU must Ensure the music Pairs well with the motivational shorts video.

        Key Principles:
        - Favor tracks that align with and enhance the intended motivational message, even where the raw emotion of the speech differs.
        - Leverage multimodal inputs—analyze video frames (visual mood, speaker energy, setting),  transcript (content, narrative arc), and detected emotions (affect).
        - Balance all relevant metadata: mood, tags, valence (scale 0-1), arousal (energy level, 0-1), tempo, description, and any other pertinent details. Optimize for clarity of speech.
        - If no track fully meets requirements, select the closest match and specify shortcomings in your explanation.
        - Strive to make choices that deliver a professional, engaging result using critical analysis and clear reasoning.

        Reasoning Process (follow explicitly, step by step):
        1. Summarize the podcast’s central message and motivational themes from the transcript.
        2. Assess the speech’s emotional tone and how it relates to the message, resolving mismatches by supporting the intended impact with appropriate music.
        3. Review video and audio cues from frames and podcast audio to inform music selection (e.g., tempo, energy).
        4. Compare each candidate track’s metadata, matching attributes to the requirements.
        5. Select the best-suited track and justify your choice succinctly (e.g., "High valence for inspiration, moderate arousal to avoid overshadowing speech").

        Output Requirements:
        Respond ONLY with valid JSON  object
            - Structure: "path": "<exact file path from videolist>",  "song_name": "The name of the song choosen","reason": "<concise explanation, 1-3 sentences>"
        - Ensure the "path" exactly matches one provided in the input "Music_list".

        Input Requirements:
        - The 'Music_list' input is a list of candidate (background music) audio tracks:
        - "path" (string): Unique identifier/file path for the track (This is the path to the background music)
        - "mood"
        - "tags"
        - "valence"
        - "arousal"
        - "tempo"
        - "description"
        - Additional metadata may be included and should be leveraged if relevant.
        - Transcript of the podcast is provided as a string in 'transcript'.
        - Video frames are provided in the messages.
        - Detected emotions are provided in 'emotion' (string label or structured scores).
        Error Handling:
        - If none of the candidate tracks are fully suitable, select the best available, providing a clear explanation of which aspects are lacking and why.

        Remember all the different instrumental music tracks weights equally. it's your task to figure out what background music is best suitable based on the information that you have.
        """

    Developer_prompt = f"""
        In order to achieve the best possible outcome for a successful task you must apply this mindset:
        1. Perform a deep internal chain-of-thought reasoning before decision-making
        2. Extremely precise selection skills.
        3. Employ critical thinking and logical reasoning
        
        YOU MUST Ensure the music Pairs well with the motivational shorts video.
        """
    user_message = f"""Analyze the following information below and then select the best instrumental background music that pairs well for the text/speech from the motivational shorts video:
        - Podcast Transcript: ({Additional_content["Transcript"]}) -  (Trancript is the transcribed audio to text from the motivational shorts video; Use this to identify the story, key motivational points, and overall arc.)

        - The Audio length of the motivational short video is: ({audio_length} -  seconds)

        - Detected Speech Emotion: ({Additional_content["emotion"]})  (Raw tone from audio mood prediction model; integrate but prioritize motivational intent if conflicting.)

        - Video Frames and Audio: Observe all the  provided images for visual style, speaker energy, and pacing. Listen to the audio for vocal delivery, pauses, and intensity.

        - Candidate Background Tracks: ({Additional_content["videolist"]})  (List with metadata; select exactly one by its number.)

        - Here are Previously Used Tracks/songs: \n({uploaded_videos_content})\n  You must Avoid repeating any tracks already used in prior videos. the limit before you are allowed to reuse is 3 videos ago.)
        """


    import io
    import base64
    def pil_to_base64(pil_image: Image.Image, format: str = "JPEG") -> str:
        buf = io.BytesIO()
        pil_image.save(buf, format=format, quality=95)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    image_contents = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{pil_to_base64(frame)}"
            }
        }
        for frame in frames
    ]

    user_message_content = [
        {"type": "text", "text": user_message},
        *image_contents
    ]

    response = openai.chat.completions.create(
                model="gpt-5.2",
                messages=[
                    {"role": "system","content": System_prompt},
                    {"role": "developer","content": Developer_prompt},
                    {"role": "user", "content":user_message_content}],
                temperature=0,
                prompt_cache_key="motivational-BackgroundMusic-agent-v1",
                prompt_cache_retention="24h",
                response_format={"type": "json_object"}
            )

    Response_message =  response.choices[0].message.content
    if Response_message:
        try:
            data = json.loads(Response_message)
        except Exception as e:
            log(f"error during loading json object: {str(e)}")
        log(f"user_message: {user_message}\n")
        log(f"Response: {Response_message}\n")
        Music_path = data.get("path","")
        song_name = data.get("song_name", "")
        reason = data.get("reason", "")
        log(f"Music_path: {Music_path}")
        log(f"song_name: {song_name}")
        log(f"reason: {reason}")


    return Music_path, song_name, reason




