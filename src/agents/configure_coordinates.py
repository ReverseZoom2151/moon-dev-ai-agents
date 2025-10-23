"""
🌙 Moon Dev's Coordinate Configuration Helper
Interactive tool to help you find the right coordinates for code_runner_agent

Usage:
1. Run this script: python src/agents/configure_coordinates.py
2. Follow the prompts to click on each location
3. The script will print out coordinates you can paste into code_runner_agent.py
"""

import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pyautogui
from termcolor import cprint

def get_mouse_position_after_countdown(location_name: str, seconds: int = 5):
    """Wait for countdown then get mouse position"""
    cprint(f"\n[*] Position your mouse over: {location_name}", "cyan")
    for i in range(seconds, 0, -1):
        print(f"  Getting position in {i} seconds...", end='\r')
        time.sleep(1)

    pos = pyautogui.position()
    cprint(f"\n[+] {location_name}: ({pos.x}, {pos.y})", "green")
    return pos.x, pos.y

def main():
    """Interactive coordinate configuration"""
    cprint("\n" + "="*60, "cyan")
    cprint("🌙 Moon Dev's Coordinate Configuration Helper", "cyan")
    cprint("="*60, "cyan")

    # Get screen size
    screen_width, screen_height = pyautogui.size()
    cprint(f"\n[*] Screen size: {screen_width}x{screen_height}", "cyan")

    cprint("\n[!] Instructions:", "yellow")
    cprint("  1. Open your code editor, terminal, and any other apps", "yellow")
    cprint("  2. When prompted, move your mouse to each location", "yellow")
    cprint("  3. Keep your mouse there until the countdown finishes", "yellow")
    cprint("  4. The script will capture each position", "yellow")

    input("\n[*] Press Enter when ready to start...")

    # Collect positions
    positions = {}

    positions['code_editor'] = get_mouse_position_after_countdown(
        "Code Editor (where you want to type code)", 5
    )

    positions['terminal'] = get_mouse_position_after_countdown(
        "Terminal (command line window)", 5
    )

    positions['ai_chat'] = get_mouse_position_after_countdown(
        "AI Chat / Composer window", 5
    )

    positions['screenshot'] = get_mouse_position_after_countdown(
        "Screenshot area (top of the area you want to capture)", 5
    )

    # Print configuration
    cprint("\n" + "="*60, "green")
    cprint("✅ Configuration Complete!", "green")
    cprint("="*60, "green")

    cprint("\n[*] Copy these values into code_runner_agent.py:", "cyan")
    cprint("-"*60, "cyan")

    print(f"""
# Custom coordinates for your setup
CODE_EDITOR_X = {positions['code_editor'][0]}
CODE_EDITOR_Y = {positions['code_editor'][1]}

TERMINAL_X = {positions['terminal'][0]}
TERMINAL_Y = {positions['terminal'][1]}

AI_CHAT_X = {positions['ai_chat'][0]}
AI_CHAT_Y = {positions['ai_chat'][1]}

SCREENSHOT_X = {positions['screenshot'][0]}
SCREENSHOT_Y = {positions['screenshot'][1]}
""")

    cprint("-"*60, "cyan")
    cprint("\n[+] Configuration saved to clipboard (if supported)", "green")

    # Try to copy to clipboard
    try:
        import pyperclip
        config_text = f"""CODE_EDITOR_X = {positions['code_editor'][0]}
CODE_EDITOR_Y = {positions['code_editor'][1]}
TERMINAL_X = {positions['terminal'][0]}
TERMINAL_Y = {positions['terminal'][1]}
AI_CHAT_X = {positions['ai_chat'][0]}
AI_CHAT_Y = {positions['ai_chat'][1]}
SCREENSHOT_X = {positions['screenshot'][0]}
SCREENSHOT_Y = {positions['screenshot'][1]}"""
        pyperclip.copy(config_text)
        cprint("[+] Copied to clipboard!", "green")
    except ImportError:
        cprint("[i] Install pyperclip for clipboard support: pip install pyperclip", "blue")

if __name__ == "__main__":
    main()
