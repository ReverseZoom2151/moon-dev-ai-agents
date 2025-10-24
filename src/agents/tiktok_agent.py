'''
This AI agent will be able to watch TikTok and pull alpha from the comment section. 

I was inspired by the dumb money guys and the social arbitrage trading style that they approach. 

Essentially, social arbitrage is getting information prior to Wall Street. Wall Street is typically very slow at getting information. 

TikTok has all the information, and in the comment section of TikTok, there are some golden opportunities. 

The problem is, if you watch TikTok all day, you're gonna go brain dead. So I don't want to watch TikTok all day, but I do want the alpha. 

This agent will be able to see the trends and connect them with different Boomer Tokens (stocks). 

## 🔎 SEARCH TERMS TO IMPROVE TIKTOK ALGORITHM 🔎
Use these search terms to train your TikTok algorithm to show more relevant content for investment insights:

### Consumer Behavior & Shopping
1. "Product hauls 2024" - People showing off recent purchases gives insight into consumer spending patterns
2. "TikTok made me buy it" - Viral products experiencing sudden demand spikes
3. "Dupes for expensive brands" - Consumer price sensitivity and alternative product markets
4. "Amazon finds under $20" - Budget consumer trends and high-value products
5. "Luxury shopping experience" - Premium consumer market insights
6. "Viral products worth the hype" - Validation of trending products
7. "Brand switching stories" - Consumer loyalty shifts
8. "Shopping regrets" - Failed products and declining brands
9. "Small business finds" - Emerging brands with growth potential
10. "Shopping hacks 2024" - Price sensitivity and consumer behavior

### Technology & Innovation
11. "Tech that changed my life" - Emerging consumer tech with high adoption potential
12. "AI tools for [specific industry]" - Industry-specific AI applications gaining traction
13. "Next big tech trend" - Early adopters showcasing emerging technologies
14. "Tech CEO news" - Leadership changes and company direction
15. "Startup success stories" - Emerging companies gaining traction
16. "Tech fails to avoid" - Products losing market share
17. "New app everyone's using" - Early app adoption trends
18. "Tech founder interviews" - Insights into company strategy
19. "Tech industry layoffs" - Early signals of company contractions
20. "Product launch reactions" - Initial consumer sentiment on new releases

### Financial & Investment
21. "Stock market analysis" - Find creators who discuss market trends and stock analysis
22. "Crypto updates daily" - Cryptocurrency discussions, news, and sentiment
23. "Financial red flags" - Consumer sentiment about economic conditions or companies
24. "Recession proof businesses" - Companies with strong economic moats
25. "Dividend stock picks" - Income investment trends
26. "Financial literacy tips" - Growing investment demographics
27. "Housing market updates" - Real estate trends and sentiment
28. "Side hustle trends 2024" - Alternative income sources growing in popularity
29. "Money mistakes to avoid" - Financial caution indicators
30. "Portfolio diversification" - Investment allocation trends

### Industry-Specific
31. "Healthcare innovation" - Medical technology and healthcare disruption
32. "Green energy solutions" - Renewable energy adoption and innovation
33. "Future of transportation" - Mobility trends and EV adoption
34. "Work from home setups" - Remote work sustainability
35. "Retail store experiences" - Brick and mortar transformation
36. "Factory automation" - Manufacturing technology trends
37. "Restaurant industry changes" - Food service trends and challenges

### Meta-Search
38. "Brands going viral" - Track which companies are gaining social momentum
39. "Products with cult following" - Items with strong consumer loyalty
40. "This company is done" - Early warnings of declining brands

🌙 Moon Dev tip: Search 3-5 terms daily and interact extensively with relevant content to rapidly train the algorithm! 💰
'''

"""
🚀 Moon Dev's TikTok Alpha Scraper
Navigates TikTok, captures video and comment screenshots for trading insights

1. Opens TikTok URL in browser
2. Detects share button position once at the start (to determine comment button position)
3. For each video:
   a. First checks if the video is LIVE by examining the URL
   b. If LIVE, skips processing and moves to next video with down arrow
   c. If not LIVE, continues with screenshot and analysis
4. For regular videos:
   a. Moves to comment button and clicks it
   b. Takes screenshot of the video and comments
   c. Analyzes screenshot with GPT-4o-mini to extract content and comments
   d. Saves analysis to CSV file for trading insights
   e. Uses double-click to ensure browser activation
   f. Presses down arrow to move to next video
5. Repeats steps 3-4 to collect data from regular videos only

The alpha is in the comments! 💰


TODO - 
- this works great, til it hits a live.... and then the double click takes us to more live.
the doublclick was to activate the screen to always scroll because it would break sometimes
but those sometimes were AFTER the lives... so the lives really kill this.... i can 
probably grab the link... 
✅ FIXED: Now detects live videos by checking URL and skips processing them entirely!
"""

# Standard library imports
import base64
import os
import random
import subprocess
import sys
import time
import traceback
import webbrowser

# Platform check - macOS only
if sys.platform != 'darwin':
    print("❌ TikTok Agent Error: This agent requires macOS")
    print("   Reason: Uses macOS-specific libraries (AppKit, Quartz)")
    print("   Platform detected: " + sys.platform)
    print()
    print("💡 This agent can only run on macOS due to Apple-specific screen automation.")
    sys.exit(1)

# Third-party imports
import AppKit
import pandas as pd
import pyautogui
import Quartz

# Standard library from imports
from datetime import datetime
from pathlib import Path

# Third-party from imports
from Quartz import CoreGraphics as CG
from termcolor import cprint

# Local from imports
from src.models import model_factory

# ===== CONFIGURATION (ADJUST THESE FOR YOUR SETUP) =====

# TikTok URL to scrape
TIKTOK_URL = "https://www.tiktok.com/foryou"  # Replace with specific TikTok page

# Screenshot save location
SCREENSHOT_DIR = Path("src/data/tiktok_agent")

# CSV file for storing analysis results
ANALYSIS_CSV = SCREENSHOT_DIR / "tiktok_analysis.csv"

# File to store live video URLs
LIVE_VIDEOS_FILE = SCREENSHOT_DIR / "live_videos.txt"

# Comment button coordinates (where to click to show comments)
COMMENT_BUTTON_X = -1895  # Adjust this to your screen
COMMENT_BUTTON_Y = -610   # Adjust this to your screen
COMMENT_BUTTON_Y_NO_SOUND = -560

# check if its a share button, return true or false
# if share button, then we use COMMENT_BUTTON_Y_NO_SOUND
SOUND_X = -1895
SOUND_Y = -369

# Share/Sound button detection area size
SHARE_DETECTION_WIDTH = 80
SHARE_DETECTION_HEIGHT = 60

# Browser area coordinates (where to click to activate browser)
BROWSER_X = -1919  # Adjust this to your screen
BROWSER_Y = -1065   # Adjust this to your screen

# Safe click coordinates specifically for live videos (to avoid opening in new tab)
LIVE_SAFE_CLICK_X = -1488  # Specific coordinates for live videos
LIVE_SAFE_CLICK_Y = -883   # Specific coordinates for live videos

# URL address bar coordinates (where to click to select the URL)
URL_BAR_X = -2211  # Coordinates for the browser address bar
URL_BAR_Y = -1491  # Coordinates for the browser address bar

# Screenshot area coordinates (video + comments)
SCREENSHOT_X = -3141
SCREENSHOT_Y = -1500
SCREENSHOT_WIDTH = 2000
SCREENSHOT_HEIGHT = 1200

# Timing settings
CLICK_PAUSE = .3         # Pause after movement before clicking
SCROLL_PAUSE = .3        # Pause between videos
ACTIVATION_PAUSE = .5   # Pause after click to ensure window activation
MOVEMENT_SPEED = 0.5      # Speed of cursor movement (seconds)
BROWSER_LOAD_WAIT = .5   # Wait for browser to load TikTok
COMMENT_LOAD_WAIT = .5  # Wait for comments to load after clicking

# Number of videos to scrape
MAX_VIDEOS = 500

# AI Model settings for screenshot analysis
MODEL_TYPE = "openai"     # Using OpenAI for text analysis
MODEL_NAME = None  # Use default from model_factory (gpt-5-2025-08-07)

# Prompt for AI analysis of screenshots
ANALYSIS_PROMPT = """The left side of the image is a TikTok video and the right side are the comments. 
Please describe the video based off of what you see and then make a list of all the comments that you see."""

# Maximum retries for screenshot capture
MAX_SCREENSHOT_RETRIES = 3
SCREENSHOT_RETRY_DELAY = 2  # seconds

# Share button detection prompt
SHARE_BUTTON_PROMPT = """Please look at this screenshot and tell me if you see a share button. 
Return only true or false. Return true if you see a share button. Return false if you don't see a share button."""

# Live video detection prompt
LIVE_VIDEO_PROMPT = """Please look at this screenshot and tell me if this is a TikTok LIVE video. 
Return only true or false. Return true if you see 'LIVE' indicators or if this appears to be a live stream. Return false if this is a regular TikTok video."""

# Directory for share button detection screenshots
SHARE_DETECTION_DIR = SCREENSHOT_DIR / "share_detection"

def get_display_bounds():
    """Get all display bounds including negative coordinates"""
    try:
        # Create array for display IDs
        max_displays = 32
        
        # Modified approach to avoid tuple interpretation error
        try:
            # Get online display list
            result = CG.CGGetOnlineDisplayList(max_displays, None, None)
            
            # Check if result is a tuple with at least 3 elements (error code, count, array)
            if not isinstance(result, tuple) or len(result) < 3:
                cprint(f"⚠️ Display list returned unexpected format: {result}", "yellow")
                return []
                
            # Extract the error code, display count and array from the tuple
            err, count, display_array = result
            
            # Check if we got a valid result
            if err != 0 or count == 0:
                cprint(f"⚠️ Error getting display list: {err}", "yellow")
                return []
                
            displays = []
            for i in range(count):
                display_id = display_array[i]
                bounds = CG.CGDisplayBounds(display_id)
                displays.append({
                    'id': display_id,
                    'x': bounds.origin.x,
                    'y': bounds.origin.y,
                    'width': bounds.size.width,
                    'height': bounds.size.height
                })
                
            return displays
            
        except Exception as e:
            cprint(f"⚠️ Error in display enumeration: {e}", "yellow")
            return []
            
    except Exception as e:
        cprint(f"⚠️ Display info error (non-critical): {e}", "yellow")
        return []

def move_mouse_cg(x, y, debug=True):
    """Move mouse using CoreGraphics with display bounds checking"""
    try:
        if debug:
            try:
                displays = get_display_bounds()
                if displays:  # Only print if we actually got display info
                    cprint("\n🖥️ Display Information:", "cyan")
                    for i, display in enumerate(displays, 1):
                        cprint(f"  ├─ Display {i}: Origin({display['x']}, {display['y']}) Size({display['width']}x{display['height']})", "cyan")
            except Exception as e:
                # Suppress this error message to reduce noise
                pass
        
        # Get current position for verification
        current = CG.CGEventGetLocation(CG.CGEventCreate(None))
        cprint(f"\n📍 Starting position: ({int(current.x)}, {int(current.y)})", "yellow")
        
        # Create smooth movement
        steps = 20
        start_x, start_y = current.x, current.y
        
        for i in range(steps + 1):
            # Calculate intermediate position
            current_x = start_x + (x - start_x) * i / steps
            current_y = start_y + (y - start_y) * i / steps
            
            # Create point for movement
            point = CG.CGPoint(current_x, current_y)
            
            # Move cursor
            CG.CGWarpMouseCursorPosition(point)
            
            # Small pause between movements
            time.sleep(MOVEMENT_SPEED / steps)
            
            if debug and i % 5 == 0:
                pos = CG.CGEventGetLocation(CG.CGEventCreate(None))
                cprint(f"  ├─ Step {i}: Moving to ({int(current_x)}, {int(current_y)}) At: ({int(pos.x)}, {int(pos.y)})", "cyan")
        
        # Final position check
        final_pos = CG.CGEventGetLocation(CG.CGEventCreate(None))
        cprint(f"📍 Final position: ({int(final_pos.x)}, {int(final_pos.y)})", "yellow")
        
        # Create and post a mouse moved event
        point = CG.CGPoint(x, y)
        event = CG.CGEventCreateMouseEvent(None, CG.kCGEventMouseMoved, point, CG.kCGMouseButtonLeft)
        CG.CGEventPost(CG.kCGHIDEventTap, event)
        
        success = abs(final_pos.x - x) <= 10 and abs(final_pos.y - y) <= 10
        if not success:
            cprint(f"⚠️ Warning: Expected ({x}, {y}) but got ({int(final_pos.x)}, {int(final_pos.y)})", "yellow")
        
        return success
        
    except Exception as e:
        cprint(f"❌ Error moving mouse: {e}", "red")
        return False

def quick_click():
    """Quick and direct click using CoreGraphics"""
    try:
        # First try CoreGraphics click
        cprint("🖱️ Attempting click...", "cyan")
        
        # Get current position
        pos = CG.CGEventGetLocation(CG.CGEventCreate(None))
        
        # Create and post mouse events
        mouse_down = CG.CGEventCreateMouseEvent(None, CG.kCGEventLeftMouseDown, pos, CG.kCGMouseButtonLeft)
        mouse_up = CG.CGEventCreateMouseEvent(None, CG.kCGEventLeftMouseUp, pos, CG.kCGMouseButtonLeft)
        
        # Click with proper timing
        time.sleep(0.3)  # Small pause before
        CG.CGEventPost(CG.kCGHIDEventTap, mouse_down)
        time.sleep(0.2)  # Hold the click
        CG.CGEventPost(CG.kCGHIDEventTap, mouse_up)
        time.sleep(0.3)  # Pause after
        
        return True
    except Exception as e:
        cprint(f"❌ Error clicking: {e}", "red")
        return False

def press_down_arrow():
    """Press down arrow key to navigate to next TikTok video"""
    try:
        cprint("⬇️ Pressing down arrow key...", "cyan")
        
        # Create down arrow key events
        down_key_down = CG.CGEventCreateKeyboardEvent(None, 0x7D, True)  # 0x7D is down arrow
        down_key_up = CG.CGEventCreateKeyboardEvent(None, 0x7D, False)
        
        # Post events
        CG.CGEventPost(CG.kCGHIDEventTap, down_key_down)
        time.sleep(0.1)
        CG.CGEventPost(CG.kCGHIDEventTap, down_key_up)
        
        cprint("✅ Down arrow pressed successfully", "green")
        return True
    except Exception as e:
        cprint(f"❌ Error pressing down arrow: {e}", "red")
        return False

def capture_screenshot(video_number):
    """Capture screenshot of TikTok video and comments"""
    for attempt in range(MAX_SCREENSHOT_RETRIES):
        try:
            cprint(f"\n📸 Screenshot attempt {attempt + 1}/{MAX_SCREENSHOT_RETRIES}...", "cyan")
            
            # Ensure screenshot directory exists
            SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
            
            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = SCREENSHOT_DIR / f"tiktok_video_{video_number:03d}_{timestamp}.png"
            
            cprint(f"📝 Screenshot will be saved to: {screenshot_path}", "cyan")
            
            # Create CGImage of specified region
            cprint(f"\n📸 Capturing TikTok screenshot...", "cyan")
            cprint(f"🎯 Capture region: x={SCREENSHOT_X}, y={SCREENSHOT_Y}, width={SCREENSHOT_WIDTH}, height={SCREENSHOT_HEIGHT}", "cyan")
            
            # Ensure we have valid coordinates
            if not all(isinstance(x, (int, float)) for x in [SCREENSHOT_X, SCREENSHOT_Y, SCREENSHOT_WIDTH, SCREENSHOT_HEIGHT]):
                raise ValueError("Invalid screenshot coordinates")
            
            region = CG.CGRectMake(
                float(SCREENSHOT_X),
                float(SCREENSHOT_Y),
                float(SCREENSHOT_WIDTH),
                float(SCREENSHOT_HEIGHT)
            )
            
            # Get screenshot
            cprint("📷 Attempting to capture window...", "cyan")
            screenshot = CG.CGWindowListCreateImage(
                region,
                CG.kCGWindowListOptionOnScreenOnly,
                CG.kCGNullWindowID,
                CG.kCGWindowImageDefault
            )
            
            if screenshot is None:
                raise Exception("Failed to capture screenshot - CGWindowListCreateImage returned None")
                
            cprint("✅ Screenshot captured successfully", "green")
            
            # Get dimensions
            width = int(CG.CGImageGetWidth(screenshot))
            height = int(CG.CGImageGetHeight(screenshot))
            
            if width == 0 or height == 0:
                raise ValueError(f"Invalid screenshot dimensions: {width}x{height}")
            
            cprint(f"📏 Screenshot dimensions: {width}x{height}", "cyan")
            
            # Create bitmap context
            colorspace = CG.CGColorSpaceCreateDeviceRGB()
            context = CG.CGBitmapContextCreate(
                None,
                width,
                height,
                8,  # bits per component
                0,  # bytes per row (0 = automatically calculated)
                colorspace,
                CG.kCGImageAlphaPremultipliedFirst | CG.kCGBitmapByteOrder32Little
            )
            
            if context is None:
                raise Exception("Failed to create bitmap context")
            
            # Draw the image to the context
            CG.CGContextDrawImage(context, CG.CGRectMake(0, 0, width, height), screenshot)
            
            # Get the image from the context
            final_image = CG.CGBitmapContextCreateImage(context)
            
            if final_image is None:
                raise Exception("Failed to create final image from context")
            
            # Create NSData from the image
            image_data = AppKit.NSBitmapImageRep.alloc().initWithCGImage_(final_image)
            png_data = image_data.representationUsingType_properties_(AppKit.NSPNGFileType, None)
            
            if png_data is None:
                raise Exception("Failed to create PNG data")
            
            # Write to file
            success = png_data.writeToFile_atomically_(str(screenshot_path), True)
            
            if not success:
                raise Exception("Failed to write PNG data to file")
            
            if not screenshot_path.exists():
                raise Exception("Screenshot file was not created")
                
            file_size = screenshot_path.stat().st_size
            if file_size == 0:
                raise Exception("Screenshot file is empty")
                
            cprint(f"✨ Screenshot saved successfully: {screenshot_path}", "green")
            cprint(f"📊 File size: {file_size} bytes", "cyan")
            
            # Add a fun Moon Dev easter egg message
            cprint(f"🌙 Moon Dev says: Alpha secured from TikTok video #{video_number}! 💰", "magenta")
            
            return str(screenshot_path)
            
        except Exception as e:
            cprint(f"❌ Error capturing screenshot (attempt {attempt + 1}): {str(e)}", "red")
            cprint("📋 Full error details:", "red")
            cprint(traceback.format_exc(), "red")
            
            if attempt < MAX_SCREENSHOT_RETRIES - 1:
                cprint(f"😴 Waiting {SCREENSHOT_RETRY_DELAY} seconds before retry...", "yellow")
                time.sleep(SCREENSHOT_RETRY_DELAY)
            else:
                cprint("❌ All screenshot attempts failed!", "red")
                return None
    
    return None

def analyze_screenshot(screenshot_path, video_number, video_url=None):
    """Analyze screenshot using GPT-4o-mini to extract video content and comments"""
    try:
        cprint(f"\n🧠 Analyzing screenshot for video #{video_number} with {MODEL_NAME}...", "cyan")
        
        # Check if file exists
        if not Path(screenshot_path).exists():
            raise FileNotFoundError(f"Screenshot file not found: {screenshot_path}")
        
        # Use the imported model_factory instance directly - it's already initialized
        cprint("🏭 Using Moon Dev's Model Factory singleton...", "cyan")
        # No need to create a new instance, model_factory is already the instance
        model = model_factory.get_model(MODEL_TYPE, MODEL_NAME)
        
        if not model:
            raise Exception(f"Failed to initialize {MODEL_TYPE} model '{MODEL_NAME}'. Check your API keys and model availability.")
        
        cprint(f"✅ Successfully initialized {MODEL_TYPE} model: {MODEL_NAME}", "green")
        
        # Encode image to base64
        cprint("🔄 Encoding screenshot to base64...", "cyan")
        with open(screenshot_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Prepare message with image
        messages = [
            {"role": "system", "content": "You are an expert at analyzing TikTok videos and comments to extract trading insights."},
            {"role": "user", "content": [
                {"type": "text", "text": ANALYSIS_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]}
        ]
        
        # Get response from model - using the correct client.chat.completions.create method
        cprint("🔄 Sending image to AI for analysis...", "cyan")
        response = model.client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=1000
        )
        
        if not response or not hasattr(response.choices[0].message, 'content') or not response.choices[0].message.content:
            raise Exception("Empty response from AI model")
        
        cprint("✅ Analysis completed successfully", "green")
        
        # Extract video description and comments from response
        analysis_text = response.choices[0].message.content
        
        # Clean up the analysis text for CSV - replace newlines with spaces and escape quotes
        cprint("🧹 Cleaning up analysis text for CSV format...", "cyan")
        
        # Replace all newlines with a space
        cleaned_analysis = analysis_text.replace('\n', ' ')
        
        # Replace multiple spaces with a single space
        cleaned_analysis = ' '.join(cleaned_analysis.split())
        
        # Escape double quotes by doubling them (CSV standard)
        cleaned_analysis = cleaned_analysis.replace('"', '""')
        
        # Add to dataframe and save to CSV
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create or load existing dataframe
        if ANALYSIS_CSV.exists():
            try:
                df = pd.read_csv(ANALYSIS_CSV)
            except Exception as e:
                cprint(f"❌ Error reading existing CSV, creating new one: {e}", "yellow")
                df = pd.DataFrame(columns=["timestamp", "video_number", "screenshot_path", "analysis", "video_url"])
        else:
            df = pd.DataFrame(columns=["timestamp", "video_number", "screenshot_path", "analysis", "video_url"])
        
        # Add new row
        new_row = {
            "timestamp": timestamp,
            "video_number": video_number,
            "screenshot_path": screenshot_path,
            "analysis": cleaned_analysis,
            "video_url": video_url if video_url else "N/A"  # Add the URL to the CSV
        }
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save to CSV with proper quoting
        try:
            df.to_csv(ANALYSIS_CSV, index=False, quoting=1)  # quoting=1 is QUOTE_ALL in csv module
            cprint(f"📊 Analysis saved to CSV: {ANALYSIS_CSV}", "green")
            cprint(f"🔗 Video URL saved: {video_url if video_url else 'N/A'}", "cyan")
        except Exception as e:
            cprint(f"❌ Error saving to CSV: {e}", "red")
            # Try to save to an alternative location as backup
            backup_path = SCREENSHOT_DIR / f"tiktok_analysis_backup_{timestamp}.csv"
            try:
                df.to_csv(backup_path, index=False, quoting=1)
                cprint(f"📊 Backup analysis saved to: {backup_path}", "yellow")
            except:
                cprint("❌ Failed to save backup CSV as well", "red")
        
        cprint(f"🌙 Moon Dev says: Alpha extracted from video #{video_number}! 💸", "magenta")
        
        # Return the original analysis for any other use
        return analysis_text
        
    except Exception as e:
        cprint(f"❌ Error analyzing screenshot: {str(e)}", "red")
        cprint("📋 Full error details:", "red")
        cprint(traceback.format_exc(), "red")
        return None

def detect_share_button():
    """Take a small screenshot of the sound/share area and use AI to detect if it's a share button"""
    try:
        cprint("\n🔍 Detecting if share button is present...", "cyan")
        
        # Ensure share detection directory exists
        SHARE_DETECTION_DIR.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = SHARE_DETECTION_DIR / f"share_detection_{timestamp}.png"
        
        # Define the region around the sound/share button - small area
        region_width = SHARE_DETECTION_WIDTH
        region_height = SHARE_DETECTION_HEIGHT
        region_x = SOUND_X - (region_width // 2)  # Center the region on SOUND_X
        region_y = SOUND_Y - (region_height // 2)  # Center the region on SOUND_Y
        
        cprint(f"📸 Capturing small area around ({SOUND_X}, {SOUND_Y}) - Size: {region_width}x{region_height}", "cyan")
        
        # Create CGImage of specified region
        region = CG.CGRectMake(
            float(region_x),
            float(region_y),
            float(region_width),
            float(region_height)
        )
        
        # Get screenshot
        cprint("📷 Attempting to capture share/sound button area...", "cyan")
        screenshot = CG.CGWindowListCreateImage(
            region,
            CG.kCGWindowListOptionOnScreenOnly,
            CG.kCGNullWindowID,
            CG.kCGWindowImageDefault
        )
        
        if screenshot is None:
            raise Exception("Failed to capture share button area screenshot")
            
        cprint("✅ Share/sound area captured successfully", "green")
        
        # Get dimensions
        width = int(CG.CGImageGetWidth(screenshot))
        height = int(CG.CGImageGetHeight(screenshot))
        
        if width == 0 or height == 0:
            raise ValueError(f"Invalid screenshot dimensions: {width}x{height}")
            
        cprint(f"📏 Captured area: {width}x{height} pixels", "green")
        
        # Create bitmap context
        colorspace = CG.CGColorSpaceCreateDeviceRGB()
        context = CG.CGBitmapContextCreate(
            None,
            width,
            height,
            8,  # bits per component
            0,  # bytes per row (0 = automatically calculated)
            colorspace,
            CG.kCGImageAlphaPremultipliedFirst | CG.kCGBitmapByteOrder32Little
        )
        
        # Draw the image to the context
        CG.CGContextDrawImage(context, CG.CGRectMake(0, 0, width, height), screenshot)
        
        # Get the image from the context
        final_image = CG.CGBitmapContextCreateImage(context)
        
        # Create NSData from the image
        image_data = AppKit.NSBitmapImageRep.alloc().initWithCGImage_(final_image)
        png_data = image_data.representationUsingType_properties_(AppKit.NSPNGFileType, None)
        
        # Write to file
        success = png_data.writeToFile_atomically_(str(screenshot_path), True)
        
        if not success:
            raise Exception("Failed to write PNG data to file")
            
        if not screenshot_path.exists():
            raise Exception("Screenshot file was not created")
            
        file_size = screenshot_path.stat().st_size
        if file_size == 0:
            raise Exception("Screenshot file is empty")
        
        cprint(f"✨ Share detection screenshot saved: {screenshot_path}", "green")
        cprint(f"📊 File size: {file_size} bytes", "cyan")
        
        # Use AI to analyze the screenshot
        cprint("🧠 Analyzing screenshot to detect share button...", "cyan")
        
        # Initialize model
        model = model_factory.get_model(MODEL_TYPE, MODEL_NAME)
        
        if not model:
            raise Exception(f"Failed to initialize {MODEL_TYPE} model")
        
        # Encode image to base64
        with open(screenshot_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Prepare message with image
        messages = [
            {"role": "system", "content": "You are an expert at analyzing TikTok interface elements. Respond with ONLY 'true' or 'false'."},
            {"role": "user", "content": [
                {"type": "text", "text": SHARE_BUTTON_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]}
        ]
        
        # Get response from model
        response = model.client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=50
        )
        
        if not response or not hasattr(response.choices[0].message, 'content'):
            raise Exception("Empty response from AI model")
        
        # Extract result
        result_text = response.choices[0].message.content.strip().lower()
        
        # Parse the result
        has_share_button = "true" in result_text
        
        # Print result with colored background
        print("\n")
        if has_share_button:
            print("\033[42m\033[30m" + "  SHARE BUTTON DETECTED: TRUE  " + "\033[0m")  # Green background with black text
        else:
            print("\033[43m\033[30m" + "  NO SHARE BUTTON DETECTED: FALSE  " + "\033[0m")  # Yellow background with black text
        print("\n")
        
        cprint(f"🔍 AI response: {result_text}", "cyan")
        
        # Add a fun Moon Dev easter egg message
        cprint(f"🌙 Moon Dev says: {'Share button detected! Using alternate comment position.' if has_share_button else 'No share button found. Using standard comment position.'} 🔍", "magenta")
        
        return has_share_button
        
    except Exception as e:
        cprint(f"❌ Error detecting share button: {str(e)}", "red")
        cprint("📋 Full error details:", "red")
        cprint(traceback.format_exc(), "red")
        # Default to False if detection fails
        return False

def copy_current_url():
    """Copy the current URL from the browser address bar using keyboard shortcuts"""
    try:
        cprint("\n🔗 Copying current URL from browser address bar...", "cyan")
        
        # Clear clipboard first to avoid getting stale content
        subprocess.run(['pbcopy'], input=b'', check=True)
        
        # First move to the URL bar
        cprint(f"🖱️ Moving to URL address bar ({URL_BAR_X}, {URL_BAR_Y})", "cyan")
        if not move_mouse_cg(URL_BAR_X, URL_BAR_Y):
            cprint("❌ Failed to move to URL address bar!", "red")
            return None
            
        # Click on the URL bar
        time.sleep(CLICK_PAUSE)
        if not quick_click():
            cprint("❌ Failed to click URL address bar!", "red")
            return None
            
        time.sleep(0.5)  # Wait for URL bar to be active
        
        # Try up to 3 times to get a valid URL
        for attempt in range(3):
            cprint(f"⌨️ URL copy attempt {attempt+1}/3...", "cyan")
            
            # Try two different approaches to select the URL
            if attempt % 2 == 0:
                # First approach: Cmd+A (Select All)
                cprint("⌨️ Using Cmd+A to select entire URL...", "cyan")
                
                try:
                    # Create command key event (Cmd key)
                    cmd_down = CG.CGEventCreateKeyboardEvent(None, 0x37, True)  # 0x37 is Command key
                    a_key_down = CG.CGEventCreateKeyboardEvent(None, 0x00, True)  # 0x00 is A key
                    a_key_up = CG.CGEventCreateKeyboardEvent(None, 0x00, False)
                    cmd_up = CG.CGEventCreateKeyboardEvent(None, 0x37, False)
                    
                    # Set flag for command key
                    CG.CGEventSetFlags(a_key_down, CG.kCGEventFlagMaskCommand)
                    CG.CGEventSetFlags(a_key_up, CG.kCGEventFlagMaskCommand)
                    
                    # Post events to select all text
                    CG.CGEventPost(CG.kCGHIDEventTap, cmd_down)
                    time.sleep(0.2)
                    CG.CGEventPost(CG.kCGHIDEventTap, a_key_down)
                    time.sleep(0.2)
                    CG.CGEventPost(CG.kCGHIDEventTap, a_key_up)
                    time.sleep(0.2)
                    CG.CGEventPost(CG.kCGHIDEventTap, cmd_up)
                    
                except Exception as e:
                    cprint(f"❌ Error during Cmd+A: {e}", "red")
            else:
                # Second approach: Cmd+L (Select URL bar)
                cprint("⌨️ Using Cmd+L to select URL bar...", "cyan")
                
                try:
                    # Create command key event (Cmd key)
                    cmd_down = CG.CGEventCreateKeyboardEvent(None, 0x37, True)  # 0x37 is Command key
                    l_key_down = CG.CGEventCreateKeyboardEvent(None, 0x25, True)  # 0x25 is L key
                    l_key_up = CG.CGEventCreateKeyboardEvent(None, 0x25, False)
                    cmd_up = CG.CGEventCreateKeyboardEvent(None, 0x37, False)
                    
                    # Set flag for command key
                    CG.CGEventSetFlags(l_key_down, CG.kCGEventFlagMaskCommand)
                    CG.CGEventSetFlags(l_key_up, CG.kCGEventFlagMaskCommand)
                    
                    # Post events to select URL bar
                    CG.CGEventPost(CG.kCGHIDEventTap, cmd_down)
                    time.sleep(0.2)
                    CG.CGEventPost(CG.kCGHIDEventTap, l_key_down)
                    time.sleep(0.2)
                    CG.CGEventPost(CG.kCGHIDEventTap, l_key_up)
                    time.sleep(0.2)
                    CG.CGEventPost(CG.kCGHIDEventTap, cmd_up)
                    
                except Exception as e:
                    cprint(f"❌ Error during Cmd+L: {e}", "red")
            
            # Pause to ensure text is selected
            time.sleep(0.5)
            
            # Press Cmd+C to copy the selected URL
            cprint("⌨️ Pressing Cmd+C to copy URL...", "cyan")
            
            try:
                # Create copy key events
                cmd_down = CG.CGEventCreateKeyboardEvent(None, 0x37, True)
                c_key_down = CG.CGEventCreateKeyboardEvent(None, 0x08, True)  # 0x08 is C key
                c_key_up = CG.CGEventCreateKeyboardEvent(None, 0x08, False)
                cmd_up = CG.CGEventCreateKeyboardEvent(None, 0x37, False)
                
                # Set flag for command key
                CG.CGEventSetFlags(c_key_down, CG.kCGEventFlagMaskCommand)
                CG.CGEventSetFlags(c_key_up, CG.kCGEventFlagMaskCommand)
                
                # Post events to copy URL
                CG.CGEventPost(CG.kCGHIDEventTap, cmd_down)
                time.sleep(0.2)
                CG.CGEventPost(CG.kCGHIDEventTap, c_key_down)
                time.sleep(0.2)
                CG.CGEventPost(CG.kCGHIDEventTap, c_key_up)
                time.sleep(0.2)
                CG.CGEventPost(CG.kCGHIDEventTap, cmd_up)
                
                # Longer pause to ensure URL is copied
                time.sleep(0.5)
                
                # Press Escape to deselect the URL (to avoid accidental typing)
                cprint("⌨️ Pressing Escape to deselect URL...", "cyan")
                esc_key_down = CG.CGEventCreateKeyboardEvent(None, 0x35, True)  # 0x35 is Escape key
                esc_key_up = CG.CGEventCreateKeyboardEvent(None, 0x35, False)
                CG.CGEventPost(CG.kCGHIDEventTap, esc_key_down)
                time.sleep(0.2)
                CG.CGEventPost(CG.kCGHIDEventTap, esc_key_up)
                
            except Exception as e:
                cprint(f"❌ Error during Cmd+C: {e}", "red")
                # Continue to clipboard check anyway
            
            # Get clipboard content
            try:
                clipboard_content = subprocess.check_output(['pbpaste'], universal_newlines=True).strip()
                
                # Validate the URL
                if clipboard_content and clipboard_content.startswith("http") and "tiktok.com" in clipboard_content:
                    cprint(f"✅ Successfully copied URL: {clipboard_content}", "green")
                    
                    # Move back to browser area
                    cprint(f"🖱️ Moving back to browser area ({BROWSER_X}, {BROWSER_Y})", "cyan")
                    move_mouse_cg(BROWSER_X, BROWSER_Y)
                    
                    # Add a fun Moon Dev easter egg message
                    cprint(f"🌙 Moon Dev says: URL captured successfully! 🔗", "magenta")
                    
                    return clipboard_content
                else:
                    cprint(f"⚠️ Attempt {attempt+1}: Invalid URL or empty clipboard: '{clipboard_content}'", "yellow")
                    # Wait before retrying
                    time.sleep(1.0)
            except Exception as e:
                cprint(f"❌ Error getting clipboard content: {e}", "red")
                time.sleep(1.0)
        
        cprint("❌ Failed to copy a valid URL after 3 attempts", "red")
        
        # Move back to browser area even if failed
        cprint(f"🖱️ Moving back to browser area ({BROWSER_X}, {BROWSER_Y})", "cyan")
        move_mouse_cg(BROWSER_X, BROWSER_Y)
        
        return None
            
    except Exception as e:
        cprint(f"❌ Error copying URL: {str(e)}", "red")
        cprint(traceback.format_exc(), "red")
        
        # Try to move back to browser area even if exception occurred
        try:
            cprint(f"🖱️ Moving back to browser area ({BROWSER_X}, {BROWSER_Y})", "cyan")
            move_mouse_cg(BROWSER_X, BROWSER_Y)
        except:
            pass
            
        return None

def is_live_video(url=None):
    """Check if the current video is a live video by examining the URL"""
    try:
        cprint("\n🔍 Checking if current video is a LIVE video...", "cyan")
        
        # Use provided URL or copy it if not provided
        if not url:
            cprint("🔗 No URL provided, attempting to copy from browser...", "cyan")
            url = copy_current_url()
            
        if not url:
            cprint("⚠️ Failed to get URL, assuming not a live video", "yellow")
            return False
            
        # Check if URL contains live indicators
        is_live = "/live" in url.lower() or "live_room_mode" in url.lower()
        
        # Print result with colored background and make it VERY visible
        print("\n")
        print("=" * 60)
        if is_live:
            print("\033[41m\033[97m" + "  🔴 LIVE VIDEO DETECTED! SKIPPING PROCESSING  ".center(56) + "\033[0m")  # Red background with white text
            cprint(f"🔗 Live URL detected: {url}", "yellow")
        else:
            print("\033[42m\033[30m" + "  ✅ REGULAR VIDEO DETECTED - PROCESSING  ".center(56) + "\033[0m")  # Green background with black text
            cprint(f"🔗 Regular video URL: {url}", "green")
        print("=" * 60)
        print("\n")
        
        # Add a fun Moon Dev easter egg message
        cprint(f"🌙 Moon Dev says: {'Live video detected! Skipping processing.' if is_live else 'Regular video detected. Will process this one!'} 🔍", "magenta")
        
        return is_live
        
    except Exception as e:
        cprint(f"❌ Error checking if video is live: {str(e)}", "red")
        cprint(traceback.format_exc(), "red")
        # Default to False if detection fails
        return False

def save_live_video_url(url, video_number):
    """Save the live video URL to a file for later processing"""
    try:
        if not url:
            cprint("⚠️ No URL provided to save", "yellow")
            return False
            
        # Ensure directory exists
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Append to file
        with open(LIVE_VIDEOS_FILE, "a") as f:
            f.write(f"{timestamp} | Video #{video_number} | {url}\n")
            
        cprint(f"✅ Saved live video URL to {LIVE_VIDEOS_FILE}", "green")
        return True
        
    except Exception as e:
        cprint(f"❌ Error saving live video URL: {str(e)}", "red")
        cprint(traceback.format_exc(), "red")
        return False

def safe_navigate_from_live(video_number):
    """Safely navigate away from a live video to the next video
    
    This function is specifically designed to handle live videos with extra care
    to avoid accidentally opening them in a new tab or triggering unwanted actions.
    """
    try:
        cprint("\n🛡️ Using safe navigation for live video...", "cyan")
        
        # Use the specific coordinates for live videos that won't trigger opening in a new tab
        cprint(f"🖱️ Moving to safe area for live videos ({LIVE_SAFE_CLICK_X}, {LIVE_SAFE_CLICK_Y})", "cyan", attrs=["bold"])
        
        # First try the dedicated safe coordinates for live videos
        if not move_mouse_cg(LIVE_SAFE_CLICK_X, LIVE_SAFE_CLICK_Y):
            cprint("⚠️ Failed to move to primary safe area, trying fallback position", "yellow")
            
            # Try a different position as fallback - far from any clickable elements
            fallback_x = BROWSER_X - 200  # Far from the edge
            fallback_y = BROWSER_Y + 200  # Far from the bottom
            
            if not move_mouse_cg(fallback_x, fallback_y):
                cprint("❌ Failed to move to fallback position!", "red")
                return False
        
        # IMPORTANT: Only click ONCE with a very gentle click
        cprint("🖱️ Performing a SINGLE gentle click to activate browser", "cyan", attrs=["bold"])
        time.sleep(0.7)  # Longer pause before clicking for live videos
        
        # Get current position
        pos = CG.CGEventGetLocation(CG.CGEventCreate(None))
        
        # Create mouse events - SINGLE CLICK ONLY
        mouse_down = CG.CGEventCreateMouseEvent(None, CG.kCGEventLeftMouseDown, pos, CG.kCGMouseButtonLeft)
        mouse_up = CG.CGEventCreateMouseEvent(None, CG.kCGEventLeftMouseUp, pos, CG.kCGMouseButtonLeft)
        
        # Post events with longer pauses
        CG.CGEventPost(CG.kCGHIDEventTap, mouse_down)
        time.sleep(0.4)  # Hold the click a bit longer
        CG.CGEventPost(CG.kCGHIDEventTap, mouse_up)
        
        # Longer pause after clicking to ensure browser is activated
        time.sleep(1.0)
        
        cprint("✅ Browser activated with single click", "green")
        
        # Press down arrow to move to next video
        cprint("⬇️ Pressing down arrow to navigate to next video...", "cyan")
        if not press_down_arrow():
            cprint("❌ Failed to press down arrow!", "red")
            # Try one more time
            time.sleep(0.5)
            if not press_down_arrow():
                cprint("❌ Failed to press down arrow on second attempt!", "red")
                return False
        
        # Wait for next video to load with a longer pause for live videos
        wait_time = SCROLL_PAUSE + 1.0
        cprint(f"⏳ Waiting {wait_time} seconds for next video to load...", "yellow")
        time.sleep(wait_time)
        
        # Add some randomness to avoid detection
        random_wait = random.uniform(0.5, 1.5)
        time.sleep(random_wait)
        
        cprint(f"🌙 Moon Dev says: Safely navigated away from live video #{video_number}! 🚀", "magenta")
        return True
        
    except Exception as e:
        cprint(f"❌ Error navigating from live video: {str(e)}", "red")
        cprint(traceback.format_exc(), "red")
        return False

def scrape_tiktok():
    """Main function to scrape TikTok videos and comments"""
    try:
        cprint("\n🚀 Moon Dev's TikTok Alpha Scraper Starting...", "cyan")
        
        # Store initial position
        initial_pos = CG.CGEventGetLocation(CG.CGEventCreate(None))
        
        # Create screenshot directory
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        SHARE_DETECTION_DIR.mkdir(parents=True, exist_ok=True)
        cprint(f"\n📁 Created screenshot directories: {SCREENSHOT_DIR} and {SHARE_DETECTION_DIR}", "green")
        
        # Open TikTok in browser
        cprint(f"\n🌐 Opening TikTok URL: {TIKTOK_URL}", "cyan")
        webbrowser.open(TIKTOK_URL)
        
        # Wait for browser to load
        cprint(f"\n⏳ Waiting {BROWSER_LOAD_WAIT} seconds for browser to load...", "yellow")
        time.sleep(BROWSER_LOAD_WAIT)
        
        # IMPORTANT: Detect share button ONCE at the start to determine comment button position
        cprint("\n🔍 Detecting share button position (one-time check)...", "cyan")
        has_share_button = detect_share_button()
        
        # Determine which comment button Y coordinate to use based on share button detection
        comment_y = COMMENT_BUTTON_Y_NO_SOUND if has_share_button else COMMENT_BUTTON_Y
        cprint(f"\n🎯 Using comment button coordinates: ({COMMENT_BUTTON_X}, {comment_y}) for all videos", "cyan")
        
        # Main scraping loop
        for video_number in range(1, MAX_VIDEOS + 1):
            try:
                cprint(f"\n🎬 Processing TikTok video #{video_number}/{MAX_VIDEOS}...", "cyan")
                
                # First get the current video URL
                cprint("\n🔗 Getting current video URL...", "cyan")
                current_url = copy_current_url()
                cprint(f"🔗 Current video URL: {current_url if current_url else 'Unknown'}", "cyan")
                
                # FIRST: Check if current video is a LIVE video BEFORE doing anything else
                cprint("\n🔍 Checking if current video is a LIVE video...", "cyan")
                is_live = is_live_video(current_url)  # Pass the URL we already copied
                
                # If it's a live video, skip processing and move to next video
                if is_live:
                    cprint("\n⏭️ LIVE video detected - skipping processing and moving to next video", "yellow", attrs=["bold"])
                    
                    # Save the live video URL if available
                    if current_url:
                        save_live_video_url(current_url, video_number)
                    
                    # Use our specialized function to safely navigate away from live videos
                    # This function handles everything: single click + down arrow
                    if not safe_navigate_from_live(video_number):
                        cprint("⚠️ Failed to safely navigate from live video, trying fallback method...", "yellow")
                        
                        # Fallback: Try to press Escape first to close any potential popups
                        try:
                            cprint("⌨️ Pressing Escape key to close any popups...", "cyan")
                            esc_key_down = CG.CGEventCreateKeyboardEvent(None, 0x35, True)  # 0x35 is Escape key
                            esc_key_up = CG.CGEventCreateKeyboardEvent(None, 0x35, False)
                            CG.CGEventPost(CG.kCGHIDEventTap, esc_key_down)
                            time.sleep(0.2)
                            CG.CGEventPost(CG.kCGHIDEventTap, esc_key_up)
                            time.sleep(0.5)
                        except Exception as e:
                            cprint(f"⚠️ Error pressing Escape: {e}", "yellow")
                        
                        # Then try a simple down arrow as last resort
                        press_down_arrow()
                        time.sleep(SCROLL_PAUSE + 0.5)
                    
                    # Check if we're still on TikTok after navigating from live video
                    # This helps detect if we accidentally opened a new tab
                    check_url = copy_current_url()
                    if check_url and "tiktok.com" in check_url:
                        cprint("✅ Successfully stayed on TikTok after live video", "green")
                    else:
                        cprint("⚠️ May have navigated away from TikTok - attempting to return", "yellow")
                        # Try to go back to TikTok
                        webbrowser.open(TIKTOK_URL)
                        time.sleep(BROWSER_LOAD_WAIT + 1.0)
                    
                    # Skip the rest of the processing for this video
                    cprint("⏩ Moving to next video...", "yellow")
                    continue
                
                # REGULAR VIDEO PROCESSING - Only happens for non-live videos
                
                # Now move to comment button and click to show comments
                cprint(f"\n🖱️ Moving to comment button ({COMMENT_BUTTON_X}, {comment_y})", "cyan")
                if not move_mouse_cg(COMMENT_BUTTON_X, comment_y):
                    cprint("❌ Failed to move to comment button!", "red")
                    continue
                    
                time.sleep(CLICK_PAUSE)
                if not quick_click():
                    cprint("❌ Failed to click comment button!", "red")
                    continue
                    
                # Wait for comments to load
                cprint(f"\n⏳ Waiting {COMMENT_LOAD_WAIT} seconds for comments to load...", "yellow")
                time.sleep(COMMENT_LOAD_WAIT)
                
                # 1. Take screenshot
                cprint(f"\n📸 Taking screenshot of video #{video_number}...", "cyan")
                screenshot_path = capture_screenshot(video_number)
                if not screenshot_path:
                    cprint(f"⚠️ Failed to capture screenshot for video #{video_number}, continuing to next video...", "yellow")
                    continue
                
                cprint(f"✅ Screenshot saved for video #{video_number}", "green")
                
                # 2. Analyze screenshot with AI and pass the video URL
                cprint(f"\n🧠 Analyzing screenshot with AI...", "cyan")
                analysis = analyze_screenshot(screenshot_path, video_number, current_url)
                if not analysis:
                    cprint(f"⚠️ Failed to analyze screenshot for video #{video_number}, but continuing...", "yellow")
                
                # 3. Click on browser area (only after first screenshot)
                if video_number == 1:
                    cprint(f"\n🖱️ Moving to browser area ({BROWSER_X}, {BROWSER_Y}) to activate it", "cyan")
                    if not move_mouse_cg(BROWSER_X, BROWSER_Y):
                        cprint("❌ Failed to move to browser area!", "red")
                        continue
                        
                    time.sleep(CLICK_PAUSE)
                    if not quick_click():
                        cprint("❌ Failed to click browser area!", "red")
                        continue
                        
                    cprint("✅ Browser area clicked successfully", "green")
                    time.sleep(ACTIVATION_PAUSE)
                
                # 4. Scroll to next video (if not the last video)
                if video_number < MAX_VIDEOS:
                    # Make sure browser is active by clicking on browser area twice
                    cprint(f"\n🎮 Activating browser window with double-click...", "cyan")
                    if not move_mouse_cg(BROWSER_X, BROWSER_Y):
                        cprint("❌ Failed to move to browser area!", "red")
                        continue
                        
                    # First click
                    time.sleep(CLICK_PAUSE)
                    if not quick_click():
                        cprint("❌ Failed to click browser area (first click)!", "red")
                        continue
                    
                    # Short pause between clicks
                    time.sleep(0.3)
                    
                    # Second click
                    if not quick_click():
                        cprint("❌ Failed to click browser area (second click)!", "red")
                        continue
                    
                    cprint("✅ Browser window activated with double-click! 🖱️🖱️", "green")
                    time.sleep(ACTIVATION_PAUSE)
                    
                    # Now press down arrow to move to next video
                    cprint(f"\n⬇️ Scrolling to next video...", "cyan")
                    if not press_down_arrow():
                        cprint("❌ Failed to press down arrow!", "red")
                        continue
                    
                    # Wait for next video to load
                    cprint(f"\n⏳ Waiting {SCROLL_PAUSE} seconds for next video to load...", "yellow")
                    time.sleep(SCROLL_PAUSE)
                    
                    # Add some randomness to avoid detection
                    random_wait = random.uniform(0.5, 2.0)
                    time.sleep(random_wait)
                
            except Exception as e:
                cprint(f"\n❌ Error processing video #{video_number}: {str(e)}", "red")
                cprint(traceback.format_exc(), "red")
                continue
        
        cprint("\n✅ TikTok scraping completed successfully!", "green")
        cprint(f"📊 Scraped {MAX_VIDEOS} TikTok videos", "green")
        cprint(f"📁 Screenshots saved to: {SCREENSHOT_DIR}", "green")
        cprint(f"📊 Analysis saved to: {ANALYSIS_CSV}", "green")
        cprint("\n🌙 Moon Dev says: All the alpha has been collected! Time to find those trading opportunities! 💰", "magenta")
        
        # Return to initial mouse position
        move_mouse_cg(int(initial_pos.x), int(initial_pos.y))
        
    except KeyboardInterrupt:
        cprint("\n👋 Scraping cancelled by user", "yellow")
        try:
            move_mouse_cg(int(initial_pos.x), int(initial_pos.y))
        except:
            pass
    except Exception as e:
        cprint(f"\n❌ Fatal error: {str(e)}", "red")
        cprint(traceback.format_exc(), "red")
        try:
            move_mouse_cg(int(initial_pos.x), int(initial_pos.y))
        except:
            pass

def find_coordinates():
    """Helper function to find screen coordinates"""
    try:
        cprint("\n🔍 Starting coordinate finder...", "cyan")
        cprint("Move your mouse to the desired position and press Ctrl+C to capture coordinates", "yellow")
        cprint("Suggested positions to find:", "yellow")
        cprint("1. Comment button coordinates", "cyan")
        cprint("2. Browser area coordinates", "cyan")
        
        while True:
            pos = CG.CGEventGetLocation(CG.CGEventCreate(None))
            x, y = int(pos.x), int(pos.y)
            
            # Clear line and print current position
            sys.stdout.write(f"\r📍 Current position: ({x}, {y})     ")
            sys.stdout.flush()
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        pos = CG.CGEventGetLocation(CG.CGEventCreate(None))
        x, y = int(pos.x), int(pos.y)
        
        cprint(f"\n\n✅ Captured coordinates: ({x}, {y})", "green")
        cprint("Copy these coordinates to the appropriate constants in the script", "yellow")
        cprint("Example for comment button:", "yellow")
        cprint(f"COMMENT_BUTTON_X = {x}", "cyan")
        cprint(f"COMMENT_BUTTON_Y = {y}", "cyan")
        cprint("Example for browser area:", "yellow")
        cprint(f"BROWSER_X = {x}", "cyan")
        cprint(f"BROWSER_Y = {y}", "cyan")

if __name__ == "__main__":
    try:
        # Check if pyobjc-framework-Quartz is installed
        try:
            import Quartz
        except ImportError:
            cprint("\n❌ Quartz framework not found. Installing required package...", "yellow")
            os.system("pip install pyobjc-framework-Quartz")
            cprint("✅ Installation complete. Please run the script again.", "green")
            sys.exit(1)
        
        # Check command line arguments
        if len(sys.argv) > 1 and sys.argv[1] == "--find-coordinates":
            find_coordinates()
        else:
            # Run the main scraping function
            scrape_tiktok()
            
    except KeyboardInterrupt:
        cprint("\n👋 Execution cancelled by user", "yellow")
    except Exception as e:
        cprint(f"\n❌ Fatal error: {str(e)}", "red")
        cprint(traceback.format_exc(), "red")



