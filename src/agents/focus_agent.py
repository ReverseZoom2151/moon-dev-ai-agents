"""
🌙 Moon Dev's Focus Agent
Built with love by Moon Dev 🚀

This agent randomly monitors speech samples and provides focus assessments.
"""

# Use local DeepSeek flag
# available free while moon dev is streaming: https://www.youtube.com/@moondevonyt
USE_LOCAL_DEEPSEEK = False

# Standard library imports
import os
import re
import sys
import tempfile
import threading
import time as time_lib

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass

# Third-party imports
import openai
import pandas as pd
import pyaudio
import requests

# Standard library from imports
from datetime import datetime, time, timedelta
from pathlib import Path
from random import randint, uniform

# Third-party from imports
from anthropic import Anthropic
from dotenv import load_dotenv
from termcolor import cprint

# Add project root to Python path for imports
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Local from imports
from src.config import *
from src.agents.base_agent import BaseAgent
from src.models.model_priority import ModelPriority

# Load environment variables from the project root
env_path = Path(project_root) / '.env'
if not env_path.exists():
    raise ValueError(f"🚨 .env file not found at {env_path}")

# Load .env file explicitly from project root
load_dotenv(dotenv_path=env_path)

# Verify key loading
cprint(f"\n🔍 Checking environment setup...", "cyan")
cprint(f"📂 Project Root: {project_root}", "cyan")
cprint(f"📝 .env Path: {env_path}", "cyan")

# Available Model Types:
# - "claude": Anthropic's Claude models
# - "groq": Groq's hosted models
# - "openai": OpenAI's GPT models
# - "gemini": Google's Gemini models
# - "deepseek": DeepSeek models
# - "ollama": Local models through Ollama

# Available Models by Type:
# OpenAI Models:
# - "gpt-4o": Latest GPT-4 Optimized (Best for complex reasoning)
# - "gpt-4o-mini": Smaller, faster GPT-4 Optimized
# - "o1": Latest O1 model - Shows reasoning process
# - "o1-mini": Smaller O1 model
# - "o3-mini": Brand new fast reasoning model

# Claude Models:
# - "claude-3-opus-20240229": Most powerful Claude
# - "claude-3-sonnet-20240229": Balanced Claude
# - "claude-3-haiku-20240307": Fast, efficient Claude

# Gemini Models:
# - "gemini-2.0-flash-exp": Next-gen multimodal
# - "gemini-1.5-flash": Fast versatile model
# - "gemini-1.5-flash-8b": High volume tasks
# - "gemini-1.5-pro": Complex reasoning tasks

# Groq Models:
# - "mixtral-8x7b-32768": Mixtral 8x7B (32k context)
# - "gemma2-9b-it": Google Gemma 2 9B
# - "llama-3.3-70b-versatile": Llama 3.3 70B
# - "llama-3.1-8b-instant": Llama 3.1 8B
# - "llama-guard-3-8b": Llama Guard 3 8B

# DeepSeek Models:
# - "deepseek-chat": Fast chat model
# - "deepseek-reasoner": Enhanced reasoning model

# Ollama Models (Local, Free):
# - "deepseek-r1": Best for complex reasoning
# - "gemma:2b": Fast and efficient for simple tasks
# - "llama3.2": Balanced model good for most tasks

# Model priority system handles model selection automatically
# Priority order: GPT-5 → Claude Sonnet 4.5 → Gemini 2.5 Pro

# Configuration for faster testing
MIN_INTERVAL_MINUTES = 2  # Less than a second
MAX_INTERVAL_MINUTES = 6  # About a second
RECORDING_DURATION = 20  # seconds
FOCUS_THRESHOLD = 8  # Minimum acceptable focus score
AUDIO_CHUNK_SIZE = 2048
SAMPLE_RATE = 16000

# Schedule settings
SCHEDULE_START = time(5, 0)  # 5:00 AM
SCHEDULE_END = time(18, 0)   # 3:00 PM

# Voice settings
VOICE_MODEL = "tts-1"
VOICE_NAME = "onyx"  # Options: alloy, echo, fable, onyx, nova, shimmer
VOICE_SPEED = 1

# Create directories
AUDIO_DIR = Path("src/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Test transcript for debugging
TEST_TRANSCRIPT = """Hey Moon Dev here, I'm working on implementing the new trading algorithm using Python. 
The RSI calculations look good but I need to optimize the moving average calculations."""

# Focus prompt optimized for all models
FOCUS_PROMPT = """You are Moon Dev's Focus AI Agent. Your task is to analyze the following transcript and rate focus.

IMPORTANT: DO NOT USE ANY MARKDOWN OR FORMATTING. RESPOND WITH PLAIN TEXT ONLY.

RESPOND WITH EXACTLY TWO LINES:
LINE 1: Just a number from 1-10 followed by '/10' (example: '8/10')
LINE 2: One encouraging sentence (no quotes)

Consider these ratings:
- Coding discussion = high focus (8-10)
- Trading analysis = high focus (8-10)
- Random chat/topics = low focus (1-4)
- Non-work discussion = low focus (1-4)

EXAMPLE RESPONSE:
8/10
Keep crushing that code, Moon Dev! Your focus is leading to amazing results.

TRANSCRIPT TO ANALYZE:
{transcript}"""

class FocusAgent(BaseAgent):
    def __init__(self):
        """Initialize the Focus Agent"""
        # Initialize BaseAgent with model_priority
        super().__init__('focus', use_model_priority=True)

        # Check if model_priority initialized successfully
        if not self.model_priority:
            raise ValueError("🚨 Model priority system not initialized!")

        cprint("🎯 Focus Agent using model_priority system with HIGH priority", "cyan")
        cprint("   Priority order: GPT-5 → Claude Sonnet 4.5 → Gemini 2.5 Pro", "cyan")
        
        # Initialize voice client
        openai_key = os.getenv("OPENAI_KEY")
        if not openai_key:
            raise ValueError("🚨 OPENAI_KEY not found in environment variables!")
        self.openai_client = openai.OpenAI(api_key=openai_key)

        
        # Initialize Anthropic for Claude models
        anthropic_key = os.getenv("ANTHROPIC_KEY")
        if not anthropic_key:
            raise ValueError("🚨 ANTHROPIC_KEY not found in environment variables!")
        self.anthropic_client = Anthropic(api_key=anthropic_key)

        cprint("✅ Using OpenAI Whisper for speech-to-text transcription", "green")
        cprint("🎯 Moon Dev's Focus Agent initialized!", "green")
        
        self.is_recording = False
        self.current_transcript = []
        
        # Add data directory path
        self.data_dir = Path(project_root) / "src" / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.focus_log_path = self.data_dir / "focus_history.csv"
        
        # Initialize focus history DataFrame if file doesn't exist
        if not self.focus_log_path.exists():
            self._create_focus_log()
            
        cprint("📊 Focus history will be logged to: " + str(self.focus_log_path), "green")
        
        self._check_schedule()
        
    def _check_schedule(self):
        """Check if current time is within scheduled hours"""
        current_time = datetime.now().time()
        if not (SCHEDULE_START <= current_time <= SCHEDULE_END):
            cprint(f"\n🌙 Moon Dev's Focus Agent is scheduled to run between {SCHEDULE_START.strftime('%I:%M %p')} and {SCHEDULE_END.strftime('%I:%M %p')}", "yellow")
            cprint("😴 Going to sleep until next scheduled time...", "yellow")
            raise SystemExit(0)
        
    def _get_random_interval(self):
        """Get random interval between MIN and MAX minutes"""
        return uniform(MIN_INTERVAL_MINUTES * 60, MAX_INTERVAL_MINUTES * 60)
        
    def record_audio(self):
        """Record audio for specified duration and transcribe with OpenAI Whisper"""
        import wave

        temp_audio_path = None
        try:
            self.is_recording = True
            self.current_transcript = []

            cprint(f"🎤 Recording audio for {RECORDING_DURATION} seconds...", "cyan")

            # Record audio to temporary WAV file
            audio = pyaudio.PyAudio()
            stream = audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=AUDIO_CHUNK_SIZE
            )

            frames = []
            start_time = time_lib.time()

            while time_lib.time() - start_time < RECORDING_DURATION:
                data = stream.read(AUDIO_CHUNK_SIZE, exception_on_overflow=False)
                frames.append(data)

            stream.stop_stream()
            stream.close()
            audio.terminate()

            # Save to temporary WAV file
            temp_audio_path = tempfile.mktemp(suffix='.wav')
            with wave.open(temp_audio_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b''.join(frames))

            cprint("🧠 Transcribing with OpenAI Whisper...", "cyan")

            # Transcribe with OpenAI Whisper
            with open(temp_audio_path, 'rb') as audio_file:
                transcript = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )

            # Store the transcript
            self.current_transcript = [transcript.text]
            cprint(f"✅ Transcription complete: {transcript.text[:100]}...", "green")

        except Exception as e:
            cprint(f"❌ Error recording/transcribing audio: {str(e)}", "red")
        finally:
            self.is_recording = False
            # Clean up temporary file
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

    def _announce(self, message, force_voice=False):
        """Announce message with optional voice"""
        try:
            cprint(f"\n🗣️ {message}", "cyan")
            
            if not force_voice:
                return
                
            # Generate speech directly to memory and play
            response = self.openai_client.audio.speech.create(
                model=VOICE_MODEL,
                voice=VOICE_NAME,
                speed=VOICE_SPEED,
                input=message
            )
            
            # Create temporary file in system temp directory
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                for chunk in response.iter_bytes():
                    temp_file.write(chunk)
                temp_path = temp_file.name

            # Play audio based on OS
            if os.name == 'posix':
                os.system(f"afplay {temp_path}")
            else:
                os.system(f"start {temp_path}")
                time_lib.sleep(5)
            
            # Cleanup temp file
            os.unlink(temp_path)
            
        except Exception as e:
            cprint(f"❌ Error in announcement: {str(e)}", "red")

    def analyze_focus(self, transcript):
        """Analyze focus level from transcript"""
        try:
            # Debug the input
            cprint(f"\n🔍 Analyzing transcript:", "cyan")
            cprint(f"  ├─ Length: {len(transcript)} chars", "cyan")
            cprint(f"  └─ Content type check: {'chicken' in transcript.lower()}", "yellow")
            
            # Use model_priority system with HIGH priority for focus analysis
            cprint("\n🧠 Getting model response using priority system...", "cyan")
            model_response, provider, model_used = self.model_priority.get_model(
                priority=ModelPriority.HIGH,
                system_prompt=FOCUS_PROMPT,
                user_content=transcript,
                temperature=AI_TEMPERATURE,
                max_tokens=AI_MAX_TOKENS
            )

            cprint(f"✨ Used model: {provider}:{model_used}", "green")

            # Extract content from ModelResponse
            response_content = model_response.content if hasattr(model_response, 'content') else str(model_response)

            # Validate response has content
            if not response_content or len(response_content.strip()) == 0:
                cprint(f"⚠️ Model {provider}:{model_used} returned empty content", "yellow")
                cprint("❌ Cannot analyze focus without AI response", "red")
                return None

            # Print raw response for debugging
            cprint(f"\n📝 Raw model response:", "magenta")
            cprint(f"══════════════════════════════", "magenta")
            cprint(response_content[:500], "yellow")  # Show first 500 chars
            cprint(f"══════════════════════════════\n", "magenta")

            # Parse the response
            lines = response_content.split('\n')
            if len(lines) >= 2:
                score_line = lines[0].strip()
                message = lines[1].strip()

                # Extract score
                score_match = re.search(r'(\d+)/10', score_line)
                if score_match:
                    score = float(score_match.group(1))
                    return score, message

            # If parsing fails, return default values
            cprint("⚠️ Couldn't parse response, using default values", "yellow")
            return 5, "Keep crushing it Moon Dev! Your focus is amazing!"
                
        except Exception as e:
            cprint(f"❌ Error analyzing focus: {str(e)}", "red")
            return 5, "Error analyzing focus, but keep going Moon Dev!"  # Always return a tuple

    def _create_focus_log(self):
        """Create empty focus history CSV"""
        df = pd.DataFrame(columns=['timestamp', 'focus_score', 'quote'])
        df.to_csv(self.focus_log_path, index=False)
        cprint("🌟 Moon Dev's Focus History log created!", "green")

    def _log_focus_data(self, score, quote):
        """Log focus data to CSV"""
        try:
            # Create new row
            new_data = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'focus_score': score,
                'quote': quote.strip('"')  # Remove quotation marks
            }
            
            # Read existing CSV
            df = pd.read_csv(self.focus_log_path)
            
            # Append new data
            df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
            
            # Save back to CSV
            df.to_csv(self.focus_log_path, index=False)
            
            cprint("📝 Focus data logged successfully!", "green")
            
        except Exception as e:
            cprint(f"❌ Error logging focus data: {str(e)}", "red")

    def process_transcript(self, transcript):
        """Process transcript and provide focus assessment"""
        # Model will be announced automatically by model_priority.get_model()
        
        # Print the transcript being sent to AI
        cprint("\n📝 Transcript being analyzed:", "cyan")
        cprint(f"══════════════════════════════", "cyan")
        cprint(transcript, "yellow")
        cprint(f"══════════════════════════════\n", "cyan")
        
        score, message = self.analyze_focus(transcript)
        
        # Log the data
        self._log_focus_data(score, message)
        
        # Determine if voice announcement needed
        needs_voice = score < FOCUS_THRESHOLD
        
        # Format message - only include score and motivational message
        formatted_message = f"{score}/10\n{message.strip()}"
        
        # Announce
        self._announce(formatted_message, force_voice=needs_voice)
        
        return score

    def run(self):
        """Main loop for random monitoring"""
        cprint("\n🎯 Moon Dev's Focus Agent starting with voice monitoring...", "cyan")
        cprint(f"⏰ Operating hours: {SCHEDULE_START.strftime('%I:%M %p')} - {SCHEDULE_END.strftime('%I:%M %p')}", "cyan")
        
        while True:
            try:
                # Check schedule before each monitoring cycle
                self._check_schedule()
                
                # Get random interval
                interval = self._get_random_interval()
                next_check = datetime.now() + timedelta(seconds=interval)
                
                # Print next check time
                cprint(f"\n⏰ Next focus check will be around {next_check.strftime('%I:%M %p')}", "cyan")
                
                # Use time_lib instead of time
                time_lib.sleep(interval)
                
                # Start recording
                #cprint("\n🎤 Recording sample...", "cyan")
                self.record_audio()
                
                # Process recording if we got something
                if self.current_transcript:
                    full_transcript = ' '.join(self.current_transcript)
                    if full_transcript.strip():
                        #cprint("\n🎯 Got transcript:", "green")
                        #cprint(f"Length: {len(full_transcript)} chars", "cyan")
                        self.process_transcript(full_transcript)
                    else:
                        cprint("⚠️ No speech detected in sample", "yellow")
                else:
                    cprint("⚠️ No transcript generated", "yellow")
                    
            except KeyboardInterrupt:
                raise
            except Exception as e:
                cprint(f"❌ Error in main loop: {str(e)}", "red")
                time_lib.sleep(5)  # Wait before retrying

if __name__ == "__main__":
    try:
        agent = FocusAgent()
        agent.run()
    except KeyboardInterrupt:
        cprint("\n👋 Focus Agent shutting down gracefully...", "yellow")
    except Exception as e:
        cprint(f"\n❌ Fatal error: {str(e)}", "red")
