from dotenv import load_dotenv
load_dotenv()   # MUST be before any core/ imports

from utils.audio_processor import process_input
from core.transcriber import transcribe_all

source = "https://youtube.com/shorts/Zmxola_HeeI?si=UOLw4Arhb9iHuP7N"
language = "hinglish" 

chunks = process_input(source)
transcript = transcribe_all(chunks, language=language)

print("\n=== Transcript ===\n")
print(transcript)