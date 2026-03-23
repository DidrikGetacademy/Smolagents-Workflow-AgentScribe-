
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
        "path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\Another love -piano.WAV",
        "song_name": "Tom Odell - Another Love | Piano cover",
        "description": "Intimate, poignant piano solo delivering raw emotional vulnerability and profound heartbreak, with gentle melodic builds and a bittersweet blend of deep melancholy, wistful longing, and subtle hopeful resolve. Valence (💖) 6.29 → bittersweet and cathartic mood. Arousal (⚡) 4.29 → tender, introspective energy. Perfect for reflective moments of loss, emotional healing, finding inner peace amid pain, vulnerable personal growth, or motivational scenes about rising stronger",
        "pairs_well_with_quotes": [
            "Because in the end, when you lose somebody, every candle, every prayer is not going to make up for the fact that the only thing that you have left is a hole in your life where that somebody that you cared about used to be.",
            "One of my favorite quotes to this day, God will put you back together in front of those who broke you.",
            "are you happy ? im not searching for happiness, im searching for peace, if there's happiness, there is sadness if something rise something must fall, if you can attain something, there's something to lose, that's why it's difficult. that's why i i stopped the moment i don't know what type of happiness you mean being satisfied with where i'm standing yes, does it mean that i don't have hard moments? i have hard moments. and that's why for me like what is it i ultimately look for? i would call it is just peace       ",
        ],
        "audio_duration_seconds": 250.0,
        "mood": {
            "melodic": 0.69,
            "romantic": 0.68,
            "film": 0.53,
            "sad": 0.73,
            "relaxing": 0.65,
            "nature": 0.64,
            "melancholic": 0.77,
            "hopeful": 0.78,
            "drama": 0.68,
            "emotional": 0.90,

        },
        "tempo": "Slow",
        "energy": {
            "💖Valence": 6.29,
            "⚡Arousal": 4.29
        },
        "genre": "Piano / Solo",
        "Tags": ["piano", "solo"]
        },

        {
        "path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\Ludovico Einaudi - Experience (Slowed) [6fBXmhBpFGE] [mp3].mp3",
        "song_name": "Ludovico Einaudi - Experience (Slowed)",
        "description": "Hypnotic minimalist piano with gradual, epic orchestral builds and meditative repetition, creating a deeply introspective journey that evolves from calm melancholy to powerful, soul-stirring inspiration. Valence (💖) 4.47 → nostalgic and bittersweet, shifting toward hopeful empowerment. Arousal (⚡) 2.96 → extremely calm and contemplative energy, perfect for slow-building motivation. Ideal for themes of personal transformation, overcoming fear, liberation, profound self-discovery, or uplifting scenes about realizing inner strength and shining brightly.",
        "pairs_well_with_quotes": [
            "Your work is going to fill a large part of your life, and the only way to be truly satisfied is to do what you believe is great work. And the only way to do great work is to love what you do. If you haven’t found it yet, keep looking. Don’t settle. As with all matters of the heart, you’ll know when you find it.",
            "Our deepest fear is not that we are inadequate. Our deepest fear is that we are powerful beyond measure. It is our light, not our darkness, that most frightens us. Your playing small does not serve the world. There is nothing enlightened about shrinking so that other people won’t feel insecure around you. We are all meant to shine as children do. It’s not just in some of us; it is in everyone. And as we let our own lights shine, we unconsciously give other people permission to do the same. As we are liberated from our own fear, our presence automatically liberates others.",
            "A good friend once said to me, I can love you and still let you go. So, Hannah, I love you.",
        ],
        "audio length":356.0,
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

        {
            "path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\Way own we go instrumental - kaleo [edit audio] [mp3].WAV",
            "song_name": "Way Down We Go Instrumental",
            "description": "This instrumental edit of Kaleo's iconic \"Way Down We Go\" distills the track's haunting essence into a concise 34-second atmospheric piece, stripping away the powerful vocals to emphasize brooding piano openings, resonant slow-building drums, and ambient layers that evoke a sense of deep introspection and inevitable descent. The slow tempo and moderate energy (low valence with subtle arousal) create a dark, dreamy soundscape infused with dramatic tension, emotional weight, and faint romantic undertones—perfect for underscoring moments of struggle, obsession, resilience, and hard-won triumph. With its contemporary pop-ambient fusion and tags highlighting slow pacing, prominent drums, and ethereal ambiance, it delivers an inspiring yet shadowy vibe that amplifies themes of pushing through doubt, hardship, and setbacks toward victory, making it an ideal backdrop for motivational montages, dramatic reflections, advertising spots, or introspective edits paired with quotes about relentless drive and the brutal reality of chasing greatness.",
            "pairs_well_with_quotes": [
                "Obsession is going to be talent every time. You got all the talent in the world, but are you obsessed? You gotta want it as bad as you want to breathe",
                "You're going to lose sleep. You'll doubt whether it'll work. You'll stress to make ends meet. You won't finish your to-do list. You'll wonder whether you made the right call and have no way to know for years. This is what hard feels like, and that's okay. Everything worth doing is hard, and the more worth doing it is, the harder it is. The greater the payoff, the greater the hardship. If it's hard, good. It means no one else will do it.",
                "Here's what you should know about winning before you chase it. Winning's not loyal to you. It doesn't care about you. Winning doesn't care how sore you are. Winning doesn't care how much sleep you get. Winning doesn't care how hard you work at times. Sometimes a guy doesn't outwork you and he still wins. It isn't fair, man. Sometimes there is no justice. Winning requires all of you and then more and it promises you nothing. It's a mastermind of creating fear and doubt in your mind. It causes setback after setback. Are you willing to sprint with a distant son? And why chase this thing called winning? Because the only thing that's guaranteed in life is no chasing is losing.",
            ],
            "audio length": 34.0,
            "mood": {
                "dream": 0.67,
                "dramatic": 0.60,
                "dark": 0.72,
                "love": 0.73,
                "advertising": 0.50,
                "romantic": 0.55,
                "inspiring": 0.60,
                "drama": 0.54,
                "emotional": 0.53
            },
            "tempo": "Slow",
            "energy": {
                "💖Valence": 4.16,
                "⚡Arousal": 4.37
            },
            "genre": "Contemporary pop / Ambient",
            "Tags": ["slow", "drums", "ambient"]
        },
        {
            "path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\Zack Hemsey - #The Way (Instrumental)  [mp3].mp3",
            "song_name": "Zack Hemsey The Way Instrumental",
            "description": "Haunting, slow-building cinematic instrumental with ambient piano, subtle orchestral layers, and dramatic undertones, evoking deep introspection, quiet resilience, and melancholic reflection on personal journeys. Valence (💖) 2.49 → dark, negative, and profoundly melancholic mood. Arousal (⚡) 2.10 → very calm, low-energy atmosphere. Ideal for contemplative motivational scenes about the curse of competence, silent success, gratitude in uncertainty, honorable perseverance, inner vulnerability, or finding one's path through hardship and quiet determination.",
            "pairs_well_with_quotes": [
                "Curse of competence. If you're good at things and have high standards, you assume that you shouldalways do well, which means that success isn't a cause for celebration, but it's the minimum levelof reasonable performance. Anything less than victory would be a failure, and victory itself becomes nothing more than acceptable. Congratulations, you might be very successful, you also might be very miserable.",
                "When you're doing well in life, make sure to do two things. Number one, keep your mouth closed. Don't tell people your plans until they materialize. And number two, be grateful for everything you have, because at any moment it can get taken away.",
                "Success is 1% luck, 2% talent, 20% being a team player, and 77% never giving up. Be honorable.",
            ],
            "audio length": 381.0,
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
            "path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\xx-intro-instrumental.WAV",
            "song_name": "The xx Intro Instrumental",
            "description": "Hypnotic, dreamy instrumental with looping guitar riffs, subtle bass, and gradual building percussion, creating atmospheric tension, quiet anticipation, and a goosebumps-inducing rush of determination. Valence (💖) 5.64 → positive, pleasant, and hopeful mood. Arousal (⚡) 3.86 → low-to-moderate energy with subtle motivational build. Ideal for reflective yet empowering scenes about self-belief, enduring hardship without help, personal responsibility, pushing through doubt and stress, or embracing that 'it's up to you' for success & Never give up.",
            "pairs_well_with_quotes": [
                "The difference between successful people and non-successful people is here. If you want to be successful, you have to change this. What makes it hard is your lack of belief that it can happen to you. The fact of it is, though, it's very doable. This decision is yours and yours alone.",
                "you're going to loose sleep you'll doubt weather it will work you will stress to make ends meet you wont finnish your to do list you will wonder weather you made the right call you will have no ways to know for years that's what hard feels like and that's okay, everything worth doing is hard. and the more worth doing it is the harder it is the greater the pay off the  the greater the hardship if it's hard good  it means no one else will do it",
                "The bottom line is no one's coming. No one. No one's coming to push you. No one's coming to tell you to turn the TV off. No one's coming to tell you to get out the door and exercise. Nobody's coming to tell you to apply for that job that you've always dreamt about. Nobody's coming to write the business plan for you. It's up to you.",
            ],
            "audio length": 60.0,
            "mood": {
            "hopeful": 0.82,
            "film": 0.79,
            "dream": 0.78,
            "romantic": 0.72,
            "calm": 0.71,
            "dramatic": 0.71,
            "action": 0.69,
            "soft": 0.68,
            "epic": 0.65,
            "documentary": 0.65,
            "drama": 0.64,
            "inspiring": 0.64,
            "movie": 0.63,
            "nature": 0.63,
            "soundscape": 0.62,
            "background": 0.61,
            "positive": 0.60,
            "emotional":0.59,
            "relaxing": 0.55
            },
            "tempo": "Slow",
            "energy": {
                "💖Valence": 5.23,
                "⚡Arousal": 3.78
            },
            "genre": "Indie / Ambient",
            "Tags": ["slow", "guitar", "drums"]
        },
        {
            "path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\Polzeni Guitar.WAV",
            "song_name": "Polozhenie Guitar Slowed",
            "description": "Haunting slowed + reverb guitar instrumental with meditative piano undertones and atmospheric depth, evoking profound melancholy, introspective loneliness, and quiet resilience amid suffering. Valence (💖) 3.58 → dark, slightly negative, and deeply melancholic mood. Arousal (⚡) 3.03 → very calm, low-intensity energy with subtle stoic empowerment. Ideal for dark motivational scenes about embracing pain as gatekeeper, pursuing the great/impossible, accepting discomfort for exceptional growth, internal conflict, or rising different and stronger through hardship.",
            "pairs_well_with_quotes": [
                "Accept the pain, smile at the pain, embrace the pain, pain is the gatekeeper of destiny, pain is there to ask you one simple question, do you really want to achieve your goals, or are you just a talk",
                "Reminder that if you want to be exceptional, you're going to be different from everyone else. That's what makes you exceptional. You can't fit in and also be exceptional. Both have discomfort. When you fit in, you have internal conflict because you're not being 100% you. When you're exceptional, you have external conflict because everyone sees you as different. Pick one.When your friends start to say, you've changed, remember it's because they don't know how to say, you've grown.",
                "Nietzsche said I know of no better life purpose than to perish in attempting to the great and impossible The fact that something seems impossible shouldn't be a reason to not pursue it That's exactly what makes it worth pursuing Where would the courage and greatness be if success was certain and there was no risk The only true failure is shrinking away from life's challenges"
            ],
            "audio length": 60.0,
            "mood": {
                "dark": 0.68,
                "slow": 0.71,
                "background": 0.67,
                "meditative": 0.70,
                "soundscape": 0.83,
                "sad": 0.63,
                "relaxing": 0.60,
                "calm": 0.58
            },
            "tempo": "Slow",
            "energy": {
                "💖Valence": 3.58,
                "⚡Arousal": 3.03
            },
            "genre": "Guitar / Solo",
            "Tags": ["guitar", "slow", "piano"]
        },
        {
            "path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\onerepublic I aint worried tiktok whistle loop - slowed reverb.mp3",
            "song_name": "Ain't Worried Whistle",
            "description": "Dreamy slowed + reverb loop of the iconic whistling melody with ambient echoes and soft nature-inspired layers, evoking carefree nonchalance, gentle release of worries, and quiet inner confidence. Valence (💖) 3.53 → slightly melancholic yet nostalgic mood with subtle positivity. Arousal (⚡) 2.97 → very calm, gentle, and meditative energy. Ideal for reflective motivational scenes about taking action despite setbacks, disappearing to build in private, letting go of overthinking and opinions, or embracing big goals with relaxed determination and freedom from anxiety.",
            "pairs_well_with_quotes": [
                "My brain can feel a certain way, but it's not going to choose how I behave all the time. I just can't let it do it anymore. I'm going to have setbacks, I know. But if I'm feeling bad, that doesn't mean I'm doing bad. That doesn't mean I am bad. That doesn't mean that I can't still take some action. Because yeah, nothing changes and nothing changes, man.",
                "If there's one message I want you to walk away with today, it's this. You have to disappear. You stop announcing every move and start building something that speaks for itself. Disappearing means doing the work. It means doing the work in the dark. It means building in private what you don't need to prove in public. It means doing the work when no one's watching. You stop worrying about what people think and start valuing what you believe. Because a life that looks good or sounds good is nothing compared to a life that feels good.",
                "Big goals create energy. They make you more creative. They attract other people. People want to be part of it. When you set big goals, you don't just achieve more, you become more.",
            ],
            "audio length":0.0,
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
            "description": "Pumping medium-tempo electronic/techno beat with prominent driving drums and intense rhythmic builds, delivering relentless energy and hype-driven motivation. Valence (💖) 6.04 → neutral-to-positive mood with uplifting determination. Arousal (⚡) 5.70 → moderate-to-high energy for dynamic action. Ideal for high-intensity motivational scenes about overcoming failure, picking yourself up, outworking laziness with relentless discipline, winners' mindset, or pushing through setbacks toward success.",
            "pairs_well_with_quotes": [
                "Why do we fall, sir? So that we can learn to pick ourselves up",
                "Lazy people do a little work and think they be winning, but winners work as hard as possible and still think they are being lazy",
                "The magic you are looking for is in the work you are avoiding"
            ],
            "audio length": 13.0,
            "mood": {
            "commercial": 0.93,
            "energetic": 0.82,
            "action": 0.76,
            "sport": 0.75,
            "documentary": 0.65,
            "fast": 0.63,
            "adventure": 0.63,
            "corporate": 0.56,
            "retro": 0.55,
            "travel": 0.55,
            "fun": 0.53
            },
            "tempo": "Medium",
            "energy": {
                "💖Valence": 6.04,
                "⚡Arousal": 5.70
            },
            "genre": "Techno / Electronic",
            "Tags": ["techno", "electronic", "drums"]
        },
        {
            "path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\Lukas Graham 7 Years (instrumental).mp3",
            "song_name": "7 Years Piano Cover",
            "description": "Heartfelt, slow melodic piano instrumental evoking profound nostalgia, life's reflective journey through ages, and bittersweet memories with subtle hopeful uplift. Valence (💖) 3.94 → neutral-to-bittersweet mood blending melancholy and optimism. Arousal (⚡) 3.22 → low-to-moderate, calm introspective energy. Ideal for motivational scenes about never giving up to keep possibilities alive, measuring success as progress from your starting point, life's milestones, personal growth through reflection, or embracing change with strong why-driven empowerment.",            "pairs_well_with_quotes": [
                "Everybody has a turn back moment. You have a moment where you can go forward or you can give up. But the thing you have to keep in mind before you give up is that if you give up, the guarantee is it will never happen. That's the guarantee of quitting, that it will never happen no way under the sun. The only way the possibility remains that it can happen is if you never give up no matter what.",
                "Success is not how far you got. Cause see you're gonna be disappointed all the time. Cause somebody always further than you. So now you'll forever be disappointed. Success ain't how far you got. Success is how far you got from where you started.",
                "See, you can do anything. You can get up any hour, read any book, take any class, make any change, develop any skill, do any discipline. I mean, you can do it all. When this how and the why, or when the why starts to grow, the how gets simple.",
            ],
            "audio length":222.0,
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
            "genre": "Piano / Solo",
            "Tags": ["piano", "solo", "classical"]
        },
        {
            "path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\Mockingbird instrumental.WAV",
            "song_name": "Mockingbird Piano Cover",
            "description": "Tender, slow piano solo radiating raw emotional vulnerability, heartfelt fatherly love, and profound regret, with nostalgic melodies blending deep melancholy, childhood longing, and subtle hopeful healing. Valence (💖) 5.59 → bittersweet positive mood with sentimental warmth. Arousal (⚡) 3.92 → low-to-moderate, gentle introspective energy. Ideal for deeply emotional motivational scenes about childhood loneliness and feeling unprotected, parental love amid struggle, figuring out life's unique puzzle through obsession, overcoming anxiety by breathing and facing fear, or healing family wounds with quiet resilience.",
            "pairs_well_with_quotes": [
                "It's either you have it or you don't. That sounds like excuses to me. I mean, you gotta figure it out. If you really have an obsession to figure it out, you will figure it out. Every puzzle is constructed differently. Everybody has a different puzzle, man, you just gotta figure out your own puzzle. ",
                "That's the thing about being alone. It's not that you feel like you don't have anybody. It's like you feel like nobody has you. And man, I know, and especially when we're young, that's so, we can't even put it into words, but we can feel that. And I remember a lot as a child feeling that way. You know, just feeling like nobody has me. ",
                " How does Keanu Reeves, uh, deal with anxiety? Breathe. Breathe, um, try to figure out why are you afraid, what does that mean, and then, um, try and, yeah, try to just be, and let not what you're afraid of define the present that you hope to be in when you go do what you're afraid of.",
            ],
            "audio length":120.0,
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
            "path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\Cinematic Adventure Epic Podcast by Infraction [No Copyright Music] # Storyteller [mp3].mp3",
            "song_name": "Cinematic Adventure Epic Podcast",
            "description": "Slow-building cinematic epic with gentle piano openings, gradual orchestral swells, and dramatic percussion, creating a profound storytelling atmosphere of reflection, perseverance, and emotional triumph. Valence (💖) 3.12 → slightly melancholic yet hopeful mood. Arousal (⚡) 3.07 → low-to-moderate energy with inspiring intensity build. Ideal for motivational scenes about never giving up despite confusion or failure, learning from mistakes for constant growth, unconditional parental support building unbreakable confidence, life's unexpected connections, or embracing the present as the ultimate gift amid uncertainty.",
            "pairs_well_with_quotes": [
                "just don't give up even if right now it doesn't make sense one day i'm very very sure when you start to see the connections and things fall into place you will maybe also realize that it couldn't have been different even if it's difficult giving up was just simply not a question was simply never an option in the way how i look at the world it's about constantly developing yourself making a mistake is not a problem making the same mistake withoutlearning from the first one that's a problem",
                "My father was really influential at a really critical time where I had a summer where I played basketball when I was like 10 or 11 years old. And here I come playing and I don't score one point the entire summer. Really? Not one. How old were you? 11, 10, 11. You're playing against other 10, 11 year olds? Uh-huh. And you didn't score once. Not one. Were you in the game? I was in the game. How'd you not score? Because I was terrible. Really? Yeah. I remember crying about it and being upset about it. My father just gave me a hug and said, listen, whether you score zero or score 60,I'm going to love you no matter what. Wow. Now that is the most important thing that you can say to a child. Because from there I was like, okay, that gives me all the confidence in the world to fail. I have the security there. But to hell with that, I'm scoring 60. Let's go. Right, right. Right. And from there I just went to work. I just stayed with it and I kept practicing, kept practicing, kept practicing.",
                "Life continues, past is past, it's never come again. The present, it's the best gift that we have in our life. The present. Because you don't know what's going to happen tomorrow. You don't know. I don't know, he don't know. So live the present.",
            ],
            "audio length":161.0,
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
            "path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\Chill Bill Whistling [Tiktok version] [mp3].mp3",
            "song_name": "Chill Bill Whistling",
            "description": "Iconic slowed whistling loop from Rob $tone's 'Chill Bill' (sampled from Twisted Nerve/Kill Bill), with playful, childlike innocence and gentle reverb echoes, evoking cheerful whimsy, lighthearted calm, and subtle underlying warmth. Valence (💖) 4.30 → slightly positive, happy, and uplifting Mood. Arousal (⚡) 2.44 → very calm, low-energy relaxation. Ideal for lighthearted motivational scenes about self-love as the foundation, embracing inner wilderness strength, feeling quietly protected and not alone, heartwarming support from loved ones or higher power, or joyful present-moment positivity.",
            "pairs_well_with_quotes": [
                "The one thing you want to do is to love and that love should begin with you. Once you love you, you love the whole world. It's easy, it's delicious to love everybody and everything.",
                "There will be times when standing alone feels too hard, too scary, and we'll doubt our ability to make our way through the uncertainty. Someone somewhere will say, don't do it. You don't have what it takes to survive the wilderness. This is when you reach deep into your wild heart and remind yourself, I am the wilderness. ",
                "you told a beautiful story yesterday about the blind woman and I would love if you would be able to share that with everyone mark and Susan had been married a few years when suddenly she began to lose her eyesight and she got really depressed and sank into a deep slow of despond but mark said listen honey I'm going to stick with you and I'm going to help you learn how to do your job blind and so they worked hard at it he went with her to the office every day showed her how to do her job as a blind woman and then he would leave for his work at an army base well one day he turned her and said Susan I'm sorry but I'm getting to work too late you're going to have to go to job on your own and she was freaked out there's no way I can ride the bus walk the streets go up the stairs go into the office building go upstairs and sit at my desk and do this on my own and he said no honey I'm going to stay with you I'm going to teach you and so they did it for a couple of weeks and then eventually said okay I think you know it well enough I've got to go to work at an earlier hour so Monday came she got on the bus walked the block walked up the stairs seamless Tuesday Wednesday Thursday it went beautifully Friday as she's getting on the bus the bus driver says you know you're a really lucky woman she says stop it I'm a blind woman he said well yes ma'am but uh every day this week that you've gotten off the bus there's a man standing on the street corner in a military uniform and he never takes his eyes off of you and when you walk down the sidewalk he's watching you carefully like a hawk you cross the street at the right time he's watching you like a hawk you go up the stairs you open the door you go in and as soon as that door closes he stands ramrod tall gives you a salute blows you a kiss and then turns to go away and all of a sudden Susan understood that's my husband he's been watching me tirelessly for all this time I'm not alone he's watching me and that's exactly what God does for us in even a more profound way and then he calls us to do that with each other",
            ],
            "audio length":57.0,
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
            "song_name": "After Hours Instrumental",
            "description": "Immersive slowed + reverb instrumental with deep nocturnal synth layers, spacey echoes, and dreamy atmospheric textures, evoking melancholic introspection, emotional depth, and a somber yet captivating late-night vibe. Valence (💖) 5.33 → positive yet bittersweet and nostalgic mood. Arousal (⚡) 4.32 → moderate, enveloping energy.  introspective emotional reflections, or dreamy lofi-style aesthetic videos.",
            "pairs_well_with_quotes": [
                "you have got to believe in yourself in your lowest moments you've got to be able to look up look in the mirror and love what you see you've got to be able to give yourself hope you must be your own greatest cheerleader you've got to be able to hug yourself and say I love you I understand it will be alright you will smile again you have to do thatfor yourself speak truth to your own personal power",
                "Don't let life fuck you up. It's yours. It's yours to drive. Get up in the morning, write down what you're going to do in a day. Be happy on your way to your job, even if you don't like it. You got to be happy on your way. Don't think you're going to get there and be happy. You carry yourself with you. You can't run. I've tried to run. And I got money to run. But you meet yourself when you get there. You meet yourself when you get there, babies.You can't run.",
                "The only reason you are not living the life that you want to live is because you day by day keep on feeding the life that you don't want to live."
            ],
            "audio length":170.0,
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
            "path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\bloody_mary.WAV",
            "song_name": "Bloody Mary",
            "description": "A haunting and empowering instrumental remix of Lady Gaga's 'Bloody Mary,' driven by pulsing techno beats, layered electronic synths, and retro-inspired melodies that evoke a dark, dreamy atmosphere with futuristic edge. This medium-tempo track builds gradual intensity with moderate arousal (5.42/10), delivering steady, controlled energy that sustains focus and drive without overwhelming the listener. Its balanced valence (5.49/10) strikes a poignant middle ground—neither overtly joyful nor deeply somber—blending mysterious introspection and resilient uplift, making it ideal for transformation arcs, gym montages, or reflective edits where adversity turns into strength. The commercial polish and emotional depth suit corporate videos, cinematic trailers, or inspirational content paired with quotes on perseverance and inner growth.",
            "pairs_well_with_quotes": [
                "The best line I've ever heard in my life.Sometimes when you're in a dark place, you think you've been buried, but actually you've been planted.",
                "Small minds discuss other people, gossip, good minds discuss events, great minds discuss ideas.",
                "You give me your best, you keep going It's heavy I know it's heavy Don't quit till you got nothing left I'm putting out of strength Then you negotiate with your body to find more strength But don't you give up on me It hurts I know it hurts, you keep going It's about how hard you can get hit And keep moving forward It's all hard from here",
            ],
            "audio length":79.0,
            "mood": {
            "retro": 0.65,
            "motivational":0.64,
            "corporate":0.63,
            "space":0.57,
            "commercial":0.57,
            "dream": 0.53
            },
            "tempo": "Medium",
            "energy": {
                "💖Valence": 5.49,
                "⚡Arousal": 5.42
            },
            "genre": "Acoustic pop",
            "Tags": ["techno", "electronic", "synth"]
        },
        {
            "path": r"C:\Users\didri\Desktop\Full-Agent-Flow_VideoEditing\Video_clips\audio\all time low (tiktok version⧸best part!) - jon bellion [edit audio].wav",
            "song_name": "All Time Low TikTok Version",
            "description": "A viral slowed + reverb TikTok edit of Jon Bellion's 'All Time Low,' focusing on the haunting 'sad part' chorus with ethereal, echoing vocals and soft, atmospheric production that creates a deeply melancholic and introspective mood. This slow-tempo track features low valence (2.17/10), evoking poignant sadness, vulnerability, and emotional weight often associated with heartbreak, nostalgia, or quiet inner struggle. Its low-to-moderate arousal (2.87/10) delivers a calm, meditative energy that flows gently without high intensity, making it perfect for reflective voiceover edits, late-night montages, tearful confessions, or profound motivational content layered with philosophical quotes on discipline, presence, and self-discovery. Widely used in emotional TikTok aesthetics for its goosebump-inducing depth and immersive reverb.",
            "pairs_well_with_quotes": [
                 "The human mind is not made to stay at one place at one time. The commitment knows it. That's why we give it. This is why we say, okay, I commit to it. Because I know that hard times will come, that I will find excuses why not to make it. That I will have days I'm not in the mood to practice it. I know these times are going to come, but this is why you do the commitment.",
                 "discipline your thoughts. Stop thinking too much. Don't go into the past. Don't go into the future. Be in the present. Why? Because only then it's possible that the mind is free, capacity is free, and now you completely immerse into whatever you are doing. And this means you have arrived.",
                 "You may have arms and legs But unless you know three things Number one, who are you and what your value is Number two, what is your purpose here in life And number three, what is your destiny when you're done here If you don't know the answers of any of those three questions You're more disabled than I",
                ],
            "audio length":65.0,
            "mood": {
            "sad": 0.82,
            "love": 0.78,
            "melancholic": 0.76,
            "meditative": 0.74,
            "romantic": 0.66,
            "soft": 0.66,
            "christmas": 0.66,
            "drama": 0.59,
            "holiday": 0.56,
            "dream": 0.55,
            "emotional": 0.52,
            "calm":0.51
            },
            "tempo": "slow",
            "energy": {
                "💖Valence": 2.17,
                "⚡Arousal": 2.87
            },
            "genre": "Contemporary dance pop",
            "Tags": ["emotional", "Deep", "Tears","Sad"]
        },
    ]
