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
from smolagents import CodeAgent, FinalAnswerTool, GoogleSearchTool, VisitWebpageTool, TransformersModel,PythonInterpreterTool
from utility.Custom_Agent_Tools import ExtractAudioFromVideo, Fetch_top_trending_youtube_videos,SpeechToTextTool_viral_agent
import gc
from googleapiclient.errors import HttpError
import torch
load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
OPENAI_APIKEY = os.getenv("OPENAI_APIKEY")
import yaml
from neon.log import log
import utility.Global_state as Global_state




def upload_tiktok_Instagram_API(hashtags,description,title,video_path,YT_channel):
    import requests
    if YT_channel == "LR_Youtube":
        Channel_name = "LR_tiktok_instagram_api_key"
    elif YT_channel == "LRS_Youtube":
        Channel_name = "LRS_tiktok_instagram_api_key"
    elif YT_channel == "MR_Youtube":
        Channel_name = "MR_tiktok_instagram_api_key"
    elif YT_channel == "LM_Youtube":
        Channel_name = "LM_tiktok_instagram_api_key"

    url = "https://api.ayrshare.com/api/post"
    api_key = os.getenv(Channel_name)
    log(f"api_key used for tiktok/instagram upload: {api_key} - channel_name: {Channel_name}")
    PostHead = ""
    title += description
    PostHead += title
    schedule_url = "https://api.ayrshare.com/api/auto-schedule/set"
    schedule_data = {
        "title": "AfterNoon Schedule",
        "schedule": ["09:00Z", "14:00Z"],
        "daysOfWeek": ["Monday", "Wednesday", "Friday"],
        "timezone": "UTC"
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    try:
            response = requests.post(schedule_url, headers=headers, json=schedule_data)
            log(f"Schedule response: {response.json()}")
    except Exception as e:
         log(f"Error during scheduling: {str(e)}")


    data = {
        "post": PostHead,
        "platforms": ["tiktok", "instagramApi"],
        "mediaUrls": video_path,
        "hashtags": hashtags,
        "isVideo": True,
        "autoSchedule": {"schedule": True, "title": schedule_data["title"]}


    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        log(f"Video uploaded successfully to TikTok and Instagram! response: {response.json()}")
    else:
        log(f"Failed to upload video. Status code: {response.status_code}, Response: {response.text}")


def Populate_Already_Uploaded(title, description, tags, categoryId, publishAt, YT_channel, video_url,subtitle_text,error,background_audio_,song_name,video_duration=None):
    already_uploaded_videos = f"Video_clips/Youtube_Upload_folder/{YT_channel}/already_uploaded.txt"
    if error == False:
        subtitle_text_list = []
        for text in subtitle_text:
             subtitle_text_list.append(text["word"])


        with open(already_uploaded_videos,"a", encoding="UTF-8") as w:
                w.write("----------------------------------")
                w.write(f"Title: {title}\n")
                w.write(f"description: {description}\n")
                w.write(f"tags: {tags}\n")
                w.write(f"categoryId: {categoryId}\n")
                w.write(f"publishAt: {publishAt}\n")
                w.write(f"Youtube Channel: {YT_channel}\n")
                w.write(f"Current_video_name: {video_url}\n")
                w.write(f"Subtitle text: {subtitle_text_list}\n")
                w.write(f"Video duration: {video_duration}\n")
                w.write (f"Background audio used: {background_audio_}\n")
                w.write(f"Song name: {song_name}\n")
                w.write("----------------------------------" + "\n")
    else:
        with open(already_uploaded_videos,"a", encoding="UTF-8") as w:
            w.write("----------------------------------" + "\n\n\n")
            w.write("###############UPLOADING FAILED!#####################\n")
            w.write(f"Title: {title}\n")
            w.write(f"description: {description}\n")
            w.write(f"tags: {tags}\n")
            w.write(f"categoryId: {categoryId}\n")
            w.write(f"publishAt: {publishAt}\n")
            w.write(f"Youtube Channel: {YT_channel}")
            w.write(f"Current_video_name: {video_url}\n")
            w.write(f"Subtitle text: {subtitle_text}\n")
            w.write(f"Video duration: {video_duration}\n")
            w.write (f"Background audio used: {background_audio_}\n")
            w.write(f"Song name: {song_name}\n")
            w.write("----------------------------------" + "\n")


SCOPES = ["https://www.googleapis.com/auth/youtube"]
def get_single_playlist_id(youtube):
    try:
          request = youtube.playlists().list(
               part="id",
               mine=True,
               maxResults=1
          )
          response = request.execute()

          items = response.get("items", [])
          log(f"response from (get_single_playlist_id): {items}")
          if items:
               item_returned = items[0]["id"]
               log(f"returned playlist_id: {item_returned}")
               return items[0]["id"]

          else:
               log("No playlist found")
               return
    except HttpError as e:
        log(f"Error retrieving playlist_id : {e}")



def get_authenticated_service(YT_channel):
    if YT_channel == "LR_Youtube":
        Client_secret = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Secrets\LR_Youtube\client_secret_849553263621-grvsfihl7lkocrt0qgs37iipuv5lbpkl.apps.googleusercontent.com.json"
    elif YT_channel == "LRS_Youtube":
        Client_secret = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Secrets\LRS_Youtube\client_secret_2_631618294192-2maahe6qccmd7naepvd54i01ur33h1js.apps.googleusercontent.com.json"
    elif YT_channel == "MR_Youtube":
        Client_secret = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Secrets\MR_Youtube\client_secret_2_69920788682-r76oarkr2q94p99k76svqacvtv66b24a.apps.googleusercontent.com.json"
    elif YT_channel == "LM_Youtube":
        Client_secret = r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Secrets\LM_Youtube\client_secret_2_460945830865-u9g1codhuh14ht71rsdcdq7ip6dgb9cq.apps.googleusercontent.com.json"
    else:
         raise ValueError(f"Error No {YT_channel} exists.")

    print(Client_secret)

    base_dir = os.path.dirname(Client_secret)
    pricle_path =  os.path.join(base_dir, 'youtube_token.pickle')
    creds = None
    if os.path.exists(pricle_path):
        with open(pricle_path, "rb") as token:
            creds = pickle.load(token)
            log(f"Loaded pickle: {creds}")
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                Client_secret, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(pricle_path, "wb") as token:
            pickle.dump(creds, token)
    return build("youtube", "v3", credentials=creds)



def Get_latestUpload_date(YT_channel: str):
    """Retrieves the latest video from the authenticated YouTube channel, including scheduled videos,
    and provides the publish date for you to know when the last video was or will be published.

    This function uses OAuth authentication to access the authenticated user's YouTube channel.
    It fetches the uploads playlist, retrieves the most recent video, and returns the video title
    along with its `publishAt` timestamp, which is the key information for scheduling or planning  the next video upload

    Args:
        YT_channel: The identifier or name of the authenticated YouTube channel. This is used
                    to locate the appropriate OAuth credentials for the channel.

    Returns:
        A formatted string containing:
        - The title of the latest video
        - The `publishAt` date/time (scheduled or actual publish date)
        This is the critical information you should use to track when the last video was/will be published.
        If no videos are found, returns a failure message.

    Example:
        >>> Get_latestUpload_date(YT_channel=YT_Channel)
        'Latest Video uploaded for the current Youtube Channel is:
            Title: How to Learn Faster
         \n PublishedAt: 2025-09-06T18:00:00Z'
    """

    youtube = get_authenticated_service(YT_channel)
    channel_request = youtube.channels().list(
         part="contentDetails",
         mine=True
    )
    channel_response = channel_request.execute()
    uploads_playlist = channel_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]


    playlist_request = youtube.playlistItems().list(
         part="snippet",
         playlistId=uploads_playlist,
         maxResults=2
    )

    playlist_response = playlist_request.execute()
    items = playlist_response.get("items",[])
    if not items:
         return "Failed getting latest publishAt"

    videos = [] # liste av videos
    for item in items: # for hver item i listen fra response
         video_id = item["snippet"]["resourceId"]["videoId"]

         video_request = youtube.videos().list(
              part="snippet,status",
              id=video_id
         )

         video_response = video_request.execute()
         video_info = video_response["items"][0]

         publish_time = video_info["status"].get(
             "publishAt",
              video_info["snippet"]["publishedAt"]
              )
         videos.append(publish_time)
    log(f"videos: {videos}")
    return videos



def upload_video(model,file_path,YT_channel,subtitle_text=None,background_audio_=None,song_name=None,video_duration=None):
    try:
      log(f"Autenchiating now..: youtube_channel: {YT_channel}")
      youtube = get_authenticated_service(YT_channel)
      log(f"auth success")
    except Exception as e:
         log(f"error during autenciation from youtube service: {str(e)}")
         return
    try:
      title, _description, hashtags, tags, categoryId, publishAt = get_automatic_data_from_agent(model,file_path)
      log(f"[get_authenticated_service]: title: {title},\n description: {_description},\n  hashtags: {hashtags},\n tags: {tags},\n categoryId: {categoryId},\n publishAt: {publishAt}\n")

    except Exception as e:
         log(f"error during running [get_automatic_data_from_agent] message: {str(e)}")
         return
    hashtag_string = ''

    for tag in hashtags:
         hashtag_string += tag  + " "
    _description += f"{hashtag_string}"
    import utility.Global_state as Global_state
    video_url = Global_state.get_current_videourl()

    body = {
        "snippet": {
            "title": title,
            "description": _description,
            "tags": tags,
            "categoryId": categoryId,
            "defaultLanguage": "en"
        },
        "status": {
            "privacyStatus": "private",
            "publishAt": publishAt,
            "selfDeclaredMadeForKids": False
        }
    }
    try:
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
                log(f"Upload progress: {int(status.progress() * 100)}%")
        log(f"Upload complete! Video ID: {response['id']}")

        playlist_id = get_single_playlist_id(youtube)
        play_list_body = {
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {
                        "kind": "youtube#video",
                        "videoId": response['id']
                    },
                    "position": 0,
                }
            }
        try:
            error = False
            playlist_response = youtube.playlistItems().insert(
                part="snippet",
                body=play_list_body
            ).execute()
            log(f"Video added to playlist {playlist_id}: {playlist_response['id']}")
            try:
                Populate_Already_Uploaded(title,_description,tags,categoryId,publishAt,YT_channel,video_url,subtitle_text,error,background_audio_,song_name=song_name,video_duration=video_duration)
            except Exception as e:
                    log(f"error writing to populate already uploaded: {str(e)}")
            truncated_audio_path = Global_state.get_current_truncated_audio_path()
            os.remove(truncated_audio_path)
        except Exception as e:
            error = True
        return YT_channel
        # try:
        #      upload_tiktok_Instagram_API(hashtags=hashtags,description=_description,title=title,video_path=file_path,YT_channel=YT_channel)
        # except Exception as e:
        #     log(f"Error during uploading to Tiktok/Instagram: {str(e)}")
    except Exception as e:
         log(f"Error during uploading: {str(e)}")





def upload_MontageClip(model,file_path,YT_channel,subtitle_text=None,background_audio_=None):
    try:
      log(f"Autenchiating now..: youtube_channel: {YT_channel}")
      youtube = get_authenticated_service(YT_channel)
      log(f"auth success")
    except Exception as e:
         log(f"error during autenciation from youtube service: {str(e)}")
         return
    try:
      title, _description, hashtags, tags, categoryId, publishAt = get_automatic_data_from_agent_montage(model,file_path,YT_channel)
      log(f"[get_authenticated_service]: title: {title}, descriptino: {_description}, tags: {tags}, publishAt: {publishAt}")

    except Exception as e:
         log(f"error during running [get_automatic_data_from_agent] message: {str(e)}")
         return

    for tag in hashtags:
         hashtag_string += tag  + " "
    _description += f" {hashtag_string}"

    video_url = None

    body = {
        "snippet": {
            "title": title,
            "description": _description,
            "tags": tags,
            "categoryId": categoryId,
            "defaultLanguage": "en"
        },
        "status": {
            "privacyStatus": "private",
            "publishAt": publishAt,
            "selfDeclaredMadeForKids": False
        }
    }
    try:
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
                log(f"Upload progress: {int(status.progress() * 100)}%")
        log(f"Upload complete! Video ID: {response['id']}")

        playlist_id = get_single_playlist_id(youtube)
        play_list_body = {
                "snippet": {
                    "playlistId": playlist_id,
                    "resourceId": {
                        "kind": "youtube#video",
                        "videoId": response['id']
                    },
                    "position": 0,
                }
            }
        try:
            error = False
            playlist_response = youtube.playlistItems().insert(
                part="snippet",
                body=play_list_body
            ).execute()
            log(f"Video added to playlist {playlist_id}: {playlist_response['id']}")

            Populate_Already_Uploaded(title,_description,tags,categoryId,publishAt,YT_channel,video_url,subtitle_text,error,background_audio_)
        except Exception as e:
            error = True
        # try:
        #      upload_tiktok_Instagram_API(hashtags=hashtags,description=_description,title=title,video_path=file_path,YT_channel=YT_channel)
        # except Exception as e:
        #     log(f"Error during uploading to Tiktok/Instagram: {str(e)}")
        #     log(f"error adding video to playlist {str(e)}")
    except Exception as e:
         log(f"Error during uploading: {str(e)}")


def get_automatic_data_from_agent_montage(model,input_video,YT_channel):

        with open((r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Prompt_templates\viral_agent_prompt.yaml"), 'r') as stream:
                    Manager_Agent_prompt_templates = yaml.safe_load(stream)
        previous_publishAt = Get_latestUpload_date(YT_channel)
        final_answer = FinalAnswerTool()
        Extract_audio = ExtractAudioFromVideo
        fetch_youtube_video_information = Fetch_top_trending_youtube_videos
        Transcriber = SpeechToTextTool_viral_agent()
        PythonInterpeter = PythonInterpreterTool()

        Google_Websearch = GoogleSearchTool()


        manager_agent  = CodeAgent(
            model=model,
            tools=[
                  final_answer,
                  Extract_audio,
                  Transcriber,
                  fetch_youtube_video_information,
                  PythonInterpeter,
                  Google_Websearch,
                  VisitWebpageTool(),
                  ],
            max_steps=4,
            verbosity_level=1,
            prompt_templates=Manager_Agent_prompt_templates,
            additional_authorized_imports=['datetime']
        )


        emojies = "⚡🧠"
        context_vars = {
               "input_video": input_video,
               "emojies": emojies,
               "previous_publishAt": previous_publishAt
            }
        user_task = "You must generate SEO-optimized metadata including: `title`, `description`, `tags`, `hashtags`, `categoryId` and `publishAt` for my video. The goal is to create SEO-optimized metadata with high viral potential by leveraging current trends and analyzing successful videos in the same category as the input video I provide you. In your final answer, you MUST use the exact key names: `title`, `description`, `tags`, `hashtags`, `publishAt`. a valid JSON object in Your final response using the `final_answer` tool."
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
        data.get("categoryId"),
        data.get("publishAt"),
    )

global test_log_list
test_log_list = []
def get_automatic_data_from_agent(model,input_video):

        import utility.Global_state as Global_state
        YT_channel = Global_state.get_current_yt_channel()
        previous_publishAt = Get_latestUpload_date(YT_channel)
        import utility.Global_state as Global_state
        with open((r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Prompt_templates\viral_agent_prompt.yaml"), 'r') as stream:
                    Manager_Agent_prompt_templates = yaml.safe_load(stream)

        final_answer = FinalAnswerTool()
        Extract_audio = ExtractAudioFromVideo
        fetch_youtube_video_information = Fetch_top_trending_youtube_videos
        Transcriber = SpeechToTextTool_viral_agent()
        PythonInterpeter = PythonInterpreterTool()

        Google_Websearch = GoogleSearchTool()


        manager_agent  = CodeAgent(
            model=model,
            tools=[
                  final_answer,
                  Extract_audio,
                  Transcriber,
                  fetch_youtube_video_information,
                  PythonInterpeter,
                  Google_Websearch,
                  VisitWebpageTool(),
                  ],
            max_steps=4,
            verbosity_level=1,
            prompt_templates=Manager_Agent_prompt_templates,
            additional_authorized_imports=['datetime']
        )

        with open(f"Video_clips/Youtube_Upload_folder/{YT_channel}/already_uploaded.txt", "r", encoding="UTF-8") as f:
            already_uploaded_list = f.read()

        emojies = "⚡🧠"
        context_vars = {
               "input_video": input_video,
               "emojies": emojies,
               "previous_publishAt": previous_publishAt,
               'Already_uploaded_list': already_uploaded_list,
               'current_video_name': Global_state.get_current_videourl()
            }
        user_task = "You must generate SEO-optimized metadata including: `title`, `description`, `tags`, `hashtags`, `categoryId` and `publishAt` for my video. The goal is to create SEO-optimized metadata with high viral potential by leveraging current trends and analyzing successful videos in the same category as the input video I provide you. In your final answer, you MUST use the exact key names: `title`, `description`, `tags`, `hashtags`, `categoryId` `publishAt`. a valid JSON object in Your final response using the `final_answer` tool."
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
        data.get("categoryId"),
        data.get("publishAt"),
    )
if __name__ == "__main__":
     from smolagents import LiteLLMModel
     gc.collect()
     torch.cuda.empty_cache()
     Global_model = LiteLLMModel(model_id="gpt-5", reasoning_effort="high" ,api_key=OPENAI_APIKEY,max_tokens=20000)
     Global_state.set_current_yt_channel("MR_Youtube")
     upload_video(model=Global_model,file_path=r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\Youtube_Upload_folder\LR_Youtube\short11_rife.mp4",subtitle_text="",YT_channel="LR_Youtube",background_audio_="lofi")
