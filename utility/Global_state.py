
#-----------------------------------#
#Get/set Functions
#-----------------------------------#
_current_audio_url: str = None
_current_video_url: str = None
_current_youtube_channel: str = None
_current_agent_saving_file: str = None
_current_truncated_audio_path: str = None
_current_global_model: str = None
_agent_saving_copy_path: str = None
count: int = 0

def get_current_count() -> int:
    return count

def get_current_audio_path() -> str:
     return _current_audio_url

def get_current_truncated_audio_path() -> str:
    return _current_truncated_audio_path

def get_current_videourl() -> str:
    return _current_video_url

def get_current_yt_channel()-> str:
     return _current_youtube_channel

def get_current_textfile() -> str:
    return _current_agent_saving_file

def get_current_global_model() -> str:
    return _current_global_model

def set_current_yt_channel(youtube_channel: str):
    global _current_youtube_channel
    _current_youtube_channel = youtube_channel

def set_current_videourl(url: str):
    global _current_video_url
    _current_video_url = url

def set_current_textfile(url: str):
    global _current_agent_saving_file
    _current_agent_saving_file = url

def set_current_audio_path(url: str):
     global _current_audio_url
     _current_audio_url = url

def set_current_count(current_count: int):
    global count
    count = current_count

def set_current_global_model(global_model):
    global _current_global_model
    _current_global_model = global_model


def set_current_truncated_audio_path(truncated_audio_path):
    global _current_truncated_audio_path
    _current_truncated_audio_path = truncated_audio_path


#---------------------------------------#
# Queue / Threads / Functions/ Variables
#---------------------------------------#
import threading
import queue
from utility.Persistent_video_Queue import PersistentVideoQueue
video_task_que = PersistentVideoQueue(r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\work_queue_folder\Video_taskQueue_backup.json")
gpu_lock = threading.Lock()
transcript_queue = queue.Queue()
count_lock = threading.Lock()
chunk_proccesed_event = threading.Event()
Montage_clip_task_Que = queue.Queue()


videolist = [
        {
            "path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\way down we go (instrumental) - kaleo [edit audio] [mp3].mp3",
            "song_name": "Way Down We Go Instrumental",
            "lut_path": '',
            "description": "A slow, dramatic, and epic track featuring prominent drums and ambient elements. Valence (💖) 3.68 → slightly negative or melancholic mood. Arousal (⚡) 3.96 → low-to-moderate energy, not very intense. Ideal for cinematic, emotional motivational, or trailer-style scenes.",
            "mood": {
                "dramatic": 0.83,
                "trailer": 0.76,
                "dark": 0.72,
                "epic": 0.72,
                "movie": 0.71,
                "advertising": 0.70,
                "film": 0.67,
                "action": 0.67,
                "documentary": 0.66,
                "inspiring": 0.65,
                "corporate": 0.64,
                "dream": 0.61,
                "drama": 0.59,
                "adventure": 0.56,
                "motivational": 0.56,
                "background": 0.55,
                "emotional": 0.53
            },
            "tempo": "Slow",
            "energy": {
                "💖Valence": 3.68,
                "⚡Arousal": 3.96
            },
            "genre": "Contemporary pop / Ambient",
            "Tags": ["slow", "drums", "ambient"]
        },
        {
            "path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\Zack Hemsey - #The Way (Instrumental)  [mp3].mp3",
             "song_name": "Zack Hemsey The Way Instrumental",
            "lut_path": '',
            "subtitle_color": '',
            "description": "Slow, ambient, and quiet track with cinematic and dramatic undertones. Valence (💖) 2.49 → negative, melancholic mood. Arousal (⚡) 2.10 → very calm, low energy. Perfect for reflective or introspective motivational  scenes.",
            "mood": {
                "movie": 0.88,
                "dark": 0.82,
                "advertising": 0.79,
                "drama": 0.75,
                "calm": 0.75,
                "documentary": 0.72,
                "uplifting": 0.68,
                "epic": 0.66,
                "nature": 0.61,
                "emotional": 0.58,
                "soft": 0.53,
                "sad": 0.50
            },
            "tempo": "Slow",
            "energy": {
                "💖Valence": 2.49,
                "⚡Arousal": 2.10
            },
            "genre": "Cinematic / Ambient",
            "Tags": ["slow", "ambient", "quiet"]
        },
        {
            "path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\The xx - Intro (Instrumental Original) [mp3].mp3",
             "song_name": "The xx Intro Instrumental",
            "lut_path": '',
            "subtitle_color": '',
            "description": "Slow, dreamy, and melodic track with guitar and drums. Valence (💖) 5.64 → positive, slightly happy/pleasant mood. Arousal (⚡) 3.86 → low-to-moderate energy. Suitable for calm, reflective, or atmospheric motivational scenes.",
            "mood": {
                "soundscape": 0.85,
                "dream": 0.78,
                "hopeful": 0.75,
                "calm": 0.74,
                "meditative": 0.73,
                "relaxing": 0.70,
                "nature": 0.69,
                "soft": 0.64,
                "melodic": 0.61,
                "background": 0.61,
                "film": 0.60,
                "space": 0.60,
                "melancholic": 0.58,
                "sad": 0.56,
                "romantic": 0.55,
                "emotional": 0.53,
                "cool": 0.53,
                "uplifting": 0.52
            },
            "tempo": "Slow",
            "energy": {
                "💖Valence": 5.64,
                "⚡Arousal": 3.86
            },
            "genre": "Indie / Ambient",
            "Tags": ["slow", "guitar", "drums"]
        },
        {
            "path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\Polozhenie Guitar - Slowed to PERFECTION # (Sigma Song) [mp3].mp3",
             "song_name": "Polozhenie Guitar Slowed",
            "lut_path": '',
            "description": "Slow, meditative guitar and piano track. Valence (💖) 3.13 → slightly negative, melancholic mood. Arousal (⚡) 2.01 → very calm, low-intensity energy. Ideal for introspective, relaxing, or emotional motivational scenes.",
            "mood": {
                "dark": 0.79,
                "slow": 0.71,
                "meditative": 0.70,
                "soundscape": 0.67,
                "sad": 0.63,
                "relaxing": 0.60,
                "calm": 0.58
            },
            "tempo": "Slow",
            "energy": {
                "💖Valence": 3.13,
                "⚡Arousal": 2.01
            },
            "genre": "Guitar / Solo",
            "Tags": ["guitar", "slow", "piano"]
        },
        {
            "path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\onerepublic I aint worried tiktok whistle loop - slowed reverb [mp3].mp3",
             "song_name": "Ain't Worried Whistle Loop",
            "lut_path": '',
            "description": "Slow, calm, and ambient track featuring soft, nature-inspired sounds. Valence (💖) 3.53 → slightly negative, melancholic mood. Arousal (⚡) 2.97 → calm, gentle energy. Suitable for meditative or relaxing backgrounds for motivational scenes .",
            "mood": {
                "calm": 0.79,
                "relaxing": 0.79,
                "soundscape": 0.77,
                "nature": 0.75,
                "documentary": 0.67,
                "meditative": 0.65,
                "soft": 0.62,
                "movie": 0.55,
                "background": 0.54
            },
            "tempo": "Slow",
            "energy": {
                "💖Valence": 3.53,
                "⚡Arousal": 2.97
            },
            "genre": "Pop / Ambient",
            "Tags": ["slow", "ambient", "quiet"]
        },
        {
            "path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\mindsetplug369 - original sound - mindsetplug369.mp3",
            "song_name": "Mindsetplug369 Original Sound",
            "lut_path": '',
            "description": "Medium tempo techno/electronic track with prominent drums. Valence (💖) 3.94 → slightly negative/neutral mood. Arousal (⚡) 3.22 → moderate energy. Energetic and uplifting, suitable for dynamic or motivating scenes.",
            "mood": {
                "uplifting": 0.85,
                "drama": 0.76,
                "hopeful": 0.75,
                "emotional": 0.73,
                "sad": 0.71,
                "happy": 0.67,
                "calm": 0.66,
                "upbeat": 0.62,
                "melancholic": 0.61,
                "dark": 0.59,
                "romantic": 0.58,
                "dream": 0.55,
                "soft": 0.55,
                "melodic": 0.54,
                "background": 0.50
            },
            "tempo": "Medium",
            "energy": {
                "💖Valence": 3.94,
                "⚡Arousal": 3.22
            },
            "genre": "Techno / Electronic",
            "Tags": ["techno", "electronic", "drums"]
        },
        {
            "path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\Lukas Graham - 7 Years (Instrumental) [mp3].mp3",
            "song_name": "7 Years Instrumental",
            "lut_path": '',
            "description": "Slow piano track with classical style. Valence (💖) 3.94 → slightly negative/neutral mood. Arousal (⚡) 3.22 → moderate energy. Uplifting and emotional, suitable for reflective or sentimental motivational  scenes.",
            "mood": {
                "uplifting": 0.85,
                "drama": 0.76,
                "hopeful": 0.75,
                "emotional": 0.73,
                "sad": 0.71,
                "happy": 0.67,
                "calm": 0.66,
                "upbeat": 0.62,
                "melancholic": 0.61,
                "dark": 0.59,
                "romantic": 0.58,
                "dream": 0.55,
                "soft": 0.55,
                "melodic": 0.54,
                "background": 0.50
            },
            "tempo": "Slow",
            "energy": {
                "💖Valence": 3.94,
                "⚡Arousal": 3.22
            },
            "genre": "Piano / Classical",
            "Tags": ["piano", "slow", "classical"]
        },
        {
            "path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\Lukas Graham - 7 Years (Instrumental) [mp3].mp3",
             "song_name": "7 Years Piano Cover",
            "lut_path": '',
            "description": "Slow, melodic piano solo. Valence (💖) 4.36 → neutral-to-positive mood. Arousal (⚡) 3.53 → low-to-moderate energy. Suitable for calm , uplifting, or reflective  motivational content.",
            "mood": {
                "children": 0.78,
                "uplifting": 0.75,
                "melodic": 0.73,
                "commercial": 0.73,
                "upbeat": 0.71,
                "hopeful": 0.71,
                "romantic": 0.64,
                "soft": 0.63,
                "sad": 0.63,
                "happy": 0.62,
                "adventure": 0.61,
                "fun": 0.61,
                "relaxing": 0.61,
                "calm": 0.60,
                "emotional": 0.58,
                "love": 0.58,
                "ballad": 0.57,
                "travel": 0.56,
                "advertising": 0.55,
                "documentary": 0.55,
                "holiday": 0.54,
                "corporate": 0.54,
                "nature": 0.53,
                "drama": 0.52
            },
            "tempo": "Slow",
            "energy": {
                "💖Valence": 4.36,
                "⚡Arousal": 3.53
            },
            "genre": "Piano / Solo",
            "Tags": ["piano", "solo", "classical"]
        },
        {
            "path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\Eminem - Mockingbird (Piano Cover) [mp3].mp3",
             "song_name": "Mockingbird Piano Cover",
            "lut_path": '',
            "description": "Slow piano solo with melodic and romantic qualities. Valence (💖) 5.59 → positive, slightly happy mood. Arousal (⚡) 3.92 → low-to-moderate energy. Ideal for emotional motivational content, heartfelt, or sentimental scenes.",
            "mood": {
                "christmas": 0.89,
                "holiday": 0.82,
                "melodic": 0.79,
                "romantic": 0.79,
                "film": 0.74,
                "hopeful": 0.73,
                "sad": 0.73,
                "soft": 0.72,
                "melancholic": 0.71,
                "love": 0.66,
                "ballad": 0.62,
                "drama": 0.58,
                "emotional": 0.57,
                "children": 0.53
            },
            "tempo": "Slow",
            "energy": {
                "💖Valence": 5.59,
                "⚡Arousal": 3.92
            },
            "genre": "Piano / Solo",
            "Tags": ["piano", "solo", "slow"]
        },
        {
            "path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\Chill Bill Whistling [Tiktok version] [mp3].mp3",
            "song_name": "Chill Bill Whistling",
            "lut_path": '',
            "description": "Slow, classical flute track with cheerful and children-friendly mood. Valence (💖) 4.30 → slightly positive mood. Arousal (⚡) 2.44 → calm, low-energy. Uplifting and playful motivational content, suitable for lighthearted or fun content.",
            "mood": {
                "children": 0.95,
                "holiday": 0.94,
                "happy": 0.88,
                "christmas": 0.88,
                "advertising": 0.78,
                "commercial": 0.74,
                "funny": 0.74,
                "documentary": 0.71,
                "uplifting": 0.67,
                "travel": 0.63,
                "calm": 0.63,
                "fun": 0.59,
                "nature": 0.59,
                "dream": 0.54,
                "corporate": 0.52
            },
            "tempo": "Slow",
            "energy": {
                "💖Valence": 4.30,
                "⚡Arousal": 2.44
            },
            "genre": "Flute / Classical / Chill",
            "Tags": ["flute", "classical", "slow"]
        },
        {
            "path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\AFTER HOURS INSTRUMENTAL (Slowed Reverb) [mp3].mp3",

            "lut_path": '',
             "song_name": "After Hours Instrumental",
            "description": "Medium tempo techno/electronic track with synths. Valence (💖) 5.33 → positive, happy mood. Arousal (⚡) 4.32 → moderate energy. Deep, spacey, and uplifting motivational content, perfect for futuristic or atmospheric scenes.",
            "mood": {
                "deep": 0.92,
                "dark": 0.81,
                "space": 0.81,
                "dream": 0.78,
                "soundscape": 0.73,
                "cool": 0.73,
                "uplifting": 0.70,
                "melodic": 0.54
            },
            "tempo": "Medium",
            "energy": {
                "💖Valence": 5.33,
                "⚡Arousal": 4.32
            },
            "genre": "Techno / Electronic",
            "Tags": ["techno", "electronic", "synth"]
        },
        {
            "path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\Cinematic Adventure Epic Podcast by Infraction [No Copyright Music] # Storyteller [mp3].mp3",
            "song_name": "Cinematic Adventure Epic Podcast",
            "lut_path": '',
            "description": "Slow, cinematic, and epic track with piano. Valence (💖) 3.12 → slightly negative mood. Arousal (⚡) 3.07 → low-to-moderate energy. Dramatic and storytelling-focused, ideal for trailers, epic scenes, or emotional motivational content.",
            "mood": {
                "epic": 0.91,
                "action": 0.87,
                "drama": 0.80,
                "film": 0.80,
                "dramatic": 0.78,
                "trailer": 0.76,
                "documentary": 0.68,
                "emotional": 0.67,
                "movie": 0.60,
                "romantic": 0.56,
                "inspiring": 0.55,
                "soundscape": 0.51
            },
            "tempo": "Slow",
            "energy": {
                "💖Valence": 3.12,
                "⚡Arousal": 3.07
            },
            "genre": "Cinematic / Ambient",
            "Tags": ["ambient", "slow", "piano"]
        },
        {
            "path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\bloody_mary.WAV",
            "song_name": "Bloody Mary",
            "lut_path": '',
            "description": "Medium tempo acoustic pop/techno track with synths. Valence (💖) 3.68 → slightly negative, melancholic mood. Arousal (⚡) 3.96 → low-to-moderate energy. Dramatic and epic, suitable for cinematic or motivational scenes.",
            "mood": {
                "dramatic": 0.83,
                "trailer": 0.76,
                "dark": 0.72,
                "epic": 0.72,
                "movie": 0.71,
                "advertising": 0.70,
                "film": 0.67,
                "action":0.67,
                "documentary": 0.66,
                "inspiring": 0.65,
                "corporate": 0.64,
                "dream": 0.61,
                "drama": 0.59,
                "adventure": 0.56,
                "motivational": 0.56,
                "background": 0.55,
                "emotional": 0.53
            },
            "tempo": "Medium",
            "energy": {
                "💖Valence": 3.68,
                "⚡Arousal": 3.96
            },
            "genre": "Acoustic pop",
            "Tags": ["techno", "electronic", "synth"]
        },
        {
            "path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\all time low (tiktok version⧸best part!) - jon bellion [edit audio].wav",
            "song_name": "All Time Low TikTok Version",
            "lut_path": '',
            "description": "Slow, dreamy, and melodic track  Valence (💖) 2.17 → positive, slightly happy/pleasant mood. Arousal (⚡) 2.87 → low-to-moderate energy. Suitable for calm, reflective, or deep emotional motivational scenes.",
            "mood": {
                "dramatic": 0.0,
                "trailer": 0.0,
                "dark": 0.0,
                "epic": 0.0,
                "movie": 0.0,
                "advertising": 0.0,
                "film": 0.0,
                "action":0.0,
                "documentary": 0.0,
                "inspiring": 0.0,
                "corporate": 0.0,
                "dream": 0.55,
                "drama":0.59,
                "adventure":0.0,
                "motivational": 0.0,
                "background": 0.0,
                "emotional": 0.52,
                "calm": 0.51,
                "meditative": 0.74,
                "melancholic":0.76,
                "love":0.78,
                "sad":0.82
            },
            "tempo": "slow",
            "energy": {
                "💖Valence": 2.17,
                "⚡Arousal": 2.87
            },
            "genre": "Contemporary dance pop",
            "Tags": ["emotional", "Deep", "Tears"]
        },
        {
            "path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\Ludovico Einaudi - Experience (Slowed) [6fBXmhBpFGE] [mp3].mp3",
            "song_name": "Ludovico Einaudi - Experience (Slowed)",
            "lut_path": '',
            "description": "",
            "mood": {
                "dramatic": 0.0,
                "trailer": 0.0,
                "dark": 0.0,
                "epic": 0.0,
                "movie": 0.0,
                "advertising": 0.0,
                "film": 0.0,
                "action":0.0,
                "documentary": 0.0,
                "inspiring": 0.0,
                "corporate": 0.0,
                "dream": 0.00,
                "drama":0.00,
                "adventure":0.0,
                "motivational": 0.0,
                "background": 0.0,
                "emotional": 0.00,
                "calm": 0.00,
                "meditative": 0.00,
                "melancholic":0.00,
                "love":0.00,
                "sad":0.00
            },
            "tempo": "slow",
            "energy": {
                "💖Valence": 0.00,
                "⚡Arousal": 0.00
            },
            "genre": "Contemporary dance pop",
            "Tags": ["emotional", "Deep", "Tears"]
        },
    ]
