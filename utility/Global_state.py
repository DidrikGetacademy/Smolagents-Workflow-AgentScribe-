
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


Music_list = [
        {
            "path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\way down we go (instrumental) - kaleo [edit audio] [mp3].WAV",
            "song_name": "Way Down We Go Instrumental",
            "lut_path": '',
            "description": "A slow, dramatic, and epic track featuring prominent drums and ambient elements. Valence (💖) 3.68 → slightly negative or melancholic mood. Arousal (⚡) 3.96 → low-to-moderate energy, not very intense. Ideal for cinematic, emotional motivational, or trailer-style scenes.",
            "Perfect for pairing with motivational lines such as:": [
                "you're going to loose sleep you'll doubt weather it will work you will stress to make ends meet you wont finnish your to do list you will wonder weather you made the right call you will have no ways to know for years that's what hard feels like and that's okay, everything worth doing is hard. and the more worth doing it is the harder it is the greater the pay off the  the greater the hardship if it's hard good  it means no one else will do it",
                "Why do we fall, sir? So that we can learn to pick ourselves up",
                "Obsession is going to be talent every time. You got all the talent in the world, but are you obsessed? You gotta want it as bad as you want to breathe",
                "You're going to lose sleep. You'll doubt whether it'll work. You'll stress to make ends meet. You won't finish your to-do list. You'll wonder whether you made the right call and have no way to know for years. This is what hard feels like, and that's okay. Everything worth doing is hard, and the more worth doing it is, the harder it is. The greater the payoff, the greater the hardship. If it's hard, good. It means no one else will do it.",
                "Lazy people do a little work and think they be winning, but winners work as hard as possible and still think they are being lazy",
                "Here's what you should know about winning before you chase it. Winning's not loyal to you. It doesn't care about you. Winning doesn't care how sore you are. Winning doesn't care how much sleep you get. Winning doesn't care how hard you work at times. Sometimes a guy doesn't outwork you and he still wins. It isn't fair, man. Sometimes there is no justice. Winning requires all of you and then more and it promises you nothing. It's a mastermind of creating fear and doubt in your mind. It causes setback after setback. Are you willing to sprint with a distant son? And why chase this thing called winning? Because the only thing that's guaranteed in life is no chasing is losing.",
            ],
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
           "Perfect for pairing with motivational lines such as:": [
                "Success is not how far you got. Cause see you're gonna be disappointed all the time. Cause somebody always further than you. So now you'll forever be disappointed. Success ain't how far you got. Success is how far you got from where you started.",
                "The difference between successful people and non-successful people is here. If you want to be successful, you have to change this. What makes it hard is your lack of belief that it can happen to you. The fact of it is, though, it's very doable. This decision is yours and yours alone.",
                "Curse of competence. If you're good at things and have high standards, you assume that you shouldalways do well, which means that success isn't a cause for celebration, but it's the minimum levelof reasonable performance. Anything less than victory would be a failure, and victory itself becomes nothing more than acceptable. Congratulations, you might be very successful, you also might be very miserable.",
                "When you're doing well in life, make sure to do two things. Number one, keep your mouth closed. Don't tell people your plans until they materialize. And number two, be grateful for everything you have, because at any moment it can get taken away.",
                "Success is 1% luck, 2% talent, 20% being a team player, and 77% never giving up. Be honorable.",
                "Everybody has a turn back moment. You have a moment where you can go forward or you can give up. But the thing you have to keep in mind before you give up is that if you give up, the guarantee is it will never happen. That's the guarantee of quitting, that it will never happen no way under the sun. The only way the possibility remains that it can happen is if you never give up no matter what.",
            ],
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
           "Perfect for pairing with motivational lines such as:": [
                "",
                "",
                "",
                "",
                "",
                "",

            ],
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
           "Perfect for pairing with motivational lines such as:": [
                "",
                "",
                "",
                "",
                "",
                "",

            ],
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
                "Perfect for pairing with motivational lines such as:": [
                "My brain can feel a certain way, but it's not going to choose how I behave all the time. I just can't let it do it anymore. I'm going to have setbacks, I know. But if I'm feeling bad, that doesn't mean I'm doing bad. That doesn't mean I am bad. That doesn't mean that I can't still take some action. Because yeah, nothing changes and nothing changes, man.",
                "If there's one message I want you to walk away with today, it's this. You have to disappear. You stop announcing every move and start building something that speaks for itself. Disappearing means doing the work. It means doing the work in the dark. It means building in private what you don't need to prove in public. It means doing the work when no one's watching. You stop worrying about what people think and start valuing what you believe. Because a life that looks good or sounds good is nothing compared to a life that feels good.",
                "The bottom line is no one's coming. No one. No one's coming to push you. No one's coming to tell you to turn the TV off. No one's coming to tell you to get out the door and exercise. Nobody's coming to tell you to apply for that job that you've always dreamt about. Nobody's coming to write the business plan for you. It's up to you.",
                "Big goals create energy. They make you more creative. They attract other people. People want to be part of it. When you set big goals, you don't just achieve more, you become more.",

            ],
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
                      "Perfect for pairing with motivational lines such as:": [
                "",
                "",
                "",
                "",
                "",
                "",

            ],
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
           "Perfect for pairing with motivational lines such as:": [
                "",
                "",
                "",
                "",
                "",
                "",

            ],
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
           "Perfect for pairing with motivational lines such as:": [
                "",
                "",
                "",
                "",
                "",
                "",

            ],
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
           "Perfect for pairing with motivational lines such as:": [
                "",
                "",
                "",
                "",
                "",
                "",

            ],
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
           "Perfect for pairing with motivational lines such as:": [
                "",
                "",
                "",
                "",
                "",
                "",

            ],
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
           "Perfect for pairing with motivational lines such as:": [
                "",
                "",
                "",
                "",
                "",
                "",

            ],
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
           "Perfect for pairing with motivational lines such as:": [
                "",
                "",
                "",
                "",
                "",
                "",

            ],
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
           "Perfect for pairing with motivational lines such as:": [
                "",
                "",
                "",
                "",
                "",
                "",

            ],
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
              "Perfect for pairing with motivational lines such as:": [
                 "",
                 "",
                 "",
                 "",
                 "",
                 "",

                ],
            "mood": {
                "dream": 0.55,
                "drama":0.59,
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
            "Tags": ["emotional", "Deep", "Tears","Sad"]
        },
        {
            "path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\Ludovico Einaudi - Experience (Slowed) [6fBXmhBpFGE] [mp3].mp3",
            "song_name": "Ludovico Einaudi - Experience (Slowed)",
            "lut_path": '',
            "description": "",
            "Perfect for pairing with motivational lines such as:": [
                "Your work is going to fill a large part of your life, and the only way to be truly satisfied is to do what you believe is great work. And the only way to do great work is to love what you do. If you haven’t found it yet, keep looking. Don’t settle. As with all matters of the heart, you’ll know when you find it.",
                "Our deepest fear is not that we are inadequate. Our deepest fear is that we are powerful beyond measure. It is our light, not our darkness, that most frightens us. Your playing small does not serve the world. There is nothing enlightened about shrinking so that other people won’t feel insecure around you. We are all meant to shine as children do. It’s not just in some of us; it is in everyone. And as we let our own lights shine, we unconsciously give other people permission to do the same. As we are liberated from our own fear, our presence automatically liberates others.",
                "",
                "",
                "",
                "",

            ],
            "mood": {
                "emotional": 0.92,
                "melancholic": 0.88,
                "sad": 0.88,
                "relaxing": 0.83,
                "hopeful": 0.75,
                "drama": 0.72,
                "romantic": 0.71,
                "soft": 0.68,
                "nature": 0.62,
                "dramatic": 0.61,
                "calm": 0.60,
                "meditative": 0.59,
                "inspiring": 0.57,
                "ballad": 0.50,
            },
            "tempo": "slow",
            "energy": {
                "💖Valence": 4.47,
                "⚡Arousal": 2.96
            },
            "genre": "Contemporary dance pop",
            "Tags": ["emotional", "Deep", "Tears","slow"]
        },
    ]
