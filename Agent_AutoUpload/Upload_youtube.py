import os
import sys
import json
REPO_ROOT = os.path.abspath('.')
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
from dotenv import load_dotenv
from smolagents import CodeAgent, FinalAnswerTool,  DuckDuckGoSearchTool, GoogleSearchTool, VisitWebpageTool, TransformersModel,PythonInterpreterTool
from Custom_Agent_Tools import ExtractAudioFromVideo, Fetch_top_trending_youtube_videos,Read_already_uploaded_video_publishedat,SpeechToTextTool_viral_agent
import gc
import torch
load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
import yaml
from log import log 
already_uploaded_videos = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\already_uploaded.txt"                
# Scopes required to upload video
SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
secret_path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Secrets"



def get_authenticated_service():
    creds = None
    pricle_path = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Secrets\youtube_token.pickle"
    if os.path.exists(pricle_path):
        with open(pricle_path, "rb") as token:
            creds = pickle.load(token)
            log(f"Loaded pickle: {creds}")
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Secrets\client_secret_849553263621-grvsfihl7lkocrt0qgs37iipuv5lbpkl.apps.googleusercontent.com.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open(pricle_path, "wb") as token:
            pickle.dump(creds, token)
    return build("youtube", "v3", credentials=creds)




def upload_video(model,file_path, category_id="22", privacy_status="private"):
    try:
      log("Autenchiating now..")
      youtube = get_authenticated_service()
      log(f"auth success")
    except Exception as e:
         log(f"error during autenciation from youtube service: {str(e)}")
         return
    try:
      title, description, hashtags, tags, time_date = get_automatic_data_from_agent(model,file_path)
      log(f"[get_authenticated_service]: title: {title}, descriptino: {description}, tags: {tags}, time_date: {time_date}")
    except Exception as e:
         log(f"error during running [get_automatic_data_from_agent] message: {str(e)}")
         return

    description += hashtags
         
    with open(already_uploaded_videos,"a", encoding="UTF-8") as w:
         w.write("----------------------------------" + "\n")
         w.write(f"Title: {title}\n")
         w.write(f"description: {description}\n")
         w.write(f"tags: {tags}\n")
         w.write(f"PublishedAt: {time_date}\n")
         w.write("----------------------------------" + "\n")
        

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": category_id,
            "defaultLanguage": "en"
        },
        "status": {
            "privacyStatus": privacy_status,
            "publishAt": time_date,
            "selfDeclaredMadeForKids": False
        }
    }

    media = MediaFileUpload(file_path, chunksize=-1, resumable=True)
    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media
    )

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Upload progress: {int(status.progress() * 100)}%")
    print(f"Upload complete! Video ID: {response['id']}")




def get_automatic_data_from_agent(model,input_video):
        #Agent Prompts
        with open((r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Prompt_templates\Viral_prompt_template.yaml"), 'r') as stream:
                    Manager_Agent_prompt_templates = yaml.safe_load(stream)
                    log(f"Loaded system prompt from : {Manager_Agent_prompt_templates}")
        
        #Tool initalization
        final_answer = FinalAnswerTool()
        Extract_audio = ExtractAudioFromVideo
        fetch_youtube_video_information = Fetch_top_trending_youtube_videos
        Transcriber = SpeechToTextTool_viral_agent()
        PythonInterpeter = PythonInterpreterTool()
        Visit_WebPage = VisitWebpageTool()
        Google_Websearch = GoogleSearchTool()
        Read_Already_UploadedVideos = Read_already_uploaded_video_publishedat


        
        manager_agent  = CodeAgent(
            model=model,
            tools=[
                  final_answer,
                  Visit_WebPage,
                  Extract_audio,
                  Transcriber,
                  PythonInterpeter,
                  fetch_youtube_video_information,
                  Google_Websearch,
                  Read_Already_UploadedVideos,
                  ], 
            max_steps=10,
            verbosity_level=4,
            prompt_templates=Manager_Agent_prompt_templates,
            stream_outputs=True,
            additional_authorized_imports=['datetime']
            
        )

        context_vars = {
               "video_path": input_video,
               "already_uploaded":already_uploaded_videos, 
             
            }       
        user_task = (
            """
                Please generate a `title`, `description`, `tags`, `hashtags`, `time_date`  for my video. The goal is to help it go viral by leveraging current trends and analyzing similar successful videos. The unique message should highlight key insights, secret strategies, or specific elements that contributed to the virality of similar content. Think of it as a short, strategic note or idea that could help this video stand out and perform exceptionally well.in your final answer Use the exact key names: `title`, `description`, `tags`, `hashtags`, `time_date`. No additional fields.
                The goal is to create content that has high viral potential by leveraging current trends 
                and analyzing successful videos in the same category. Include a unique message that highlights 
                key insights, secret strategies, or specific elements that contributed to the virality of similar content. 
                This unique message should help the video stand out and perform exceptionally well. 
                first use your 1.step  executing code for this 2 tools `ExtractAudioFromVideo` and `transcriber` to get the text you will analyze
                Your final response with `final_answer` tool must be a valid JSON object with ONLY these exact keys: 
                `title`, `description`, `tags`, and `time_date`. 
                `time_date` must be in RFC 3339 UTC format (e.g., '2025-06-23T14:30:00Z'). 
                Do NOT include any other text or fields outside the JSON in your `final_answer`
            """
        )
        try:
            Response = manager_agent.run(
                task=user_task,
                additional_args=context_vars
            )
        except Exception as e:
             log(f"Error during agent run: {str(e)}")
        print(Response)
        import json 
        try:
             data = Response if isinstance(Response, dict) else json.loads(Response)
        except Exception as e:
             raise ValueError(f"agent output is not valid json...!")
        torch.cuda.empty_cache()
        gc.collect()

        return (
        data.get("title"),
        data.get("description"),
        data.get("hashtags"),
        data.get("tags"),
        data.get("time_date"),
    )




if __name__ == "__main__":
    from smolagents import TransformersModel
    import gc
    import torch 
    gc.collect()
    torch.cuda.empty_cache()
    print(f"SERPAPI_API_KEY: {SERPAPI_API_KEY}")
    model = TransformersModel(
            model_id = r"C:\Users\didri\Desktop\LLM-models\LLM-Models\Qwen\Qwen2.5-Coder-7B-Instruct",
            load_in_4bit=True,
            device_map="auto",
            torch_dtype="auto",
            do_sample=False,
            max_new_tokens=10000,
            use_flash_attn=True
    )
    try:
        upload_video(model=model,file_path=r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\klipp(1).mp4")
    except Exception as e:
         log(f"error")
         gc.collect()
         torch.cuda.empty_cache()


