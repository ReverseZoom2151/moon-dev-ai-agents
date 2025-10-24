'''
i run a local server with deepseek-r1, below is how to call it from your code

the deepseek-r1 model is available free while moon dev is streaming: https://www.youtube.com/@moondevonyt
'''

# Standard library imports
import sys

# Third-party imports
import requests

# Third-party from imports
from openai import OpenAI

# Easy to modify prompt at the top
PROMPT = """

build me a volume spike backtest in python
"""

# Easy to modify IP at the top
# this changes daily, get inside discord to know when it changes: https://algotradecamp.com
# check the free api keys section of the discord
LAMBDA_IP = "192.222.57.184"  # Just update this when you start a new instance

def check_server_connection(base_url: str, timeout: int = 5) -> bool:
    """Check if the server is accessible"""
    try:
        response = requests.get(f"{base_url.replace('/v1', '')}/health", timeout=timeout)
        return response.status_code == 200
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False

if __name__ == '__main__':
    base_url = f"http://{LAMBDA_IP}:8000/v1"

    # Check if server is accessible
    print(f"🔍 Checking connection to {LAMBDA_IP}...")
    if not check_server_connection(base_url):
        print(f"\n❌ ERROR: Cannot connect to server at {LAMBDA_IP}:8000")
        print(f"\n💡 Possible solutions:")
        print(f"   1. Check if the IP address is correct (currently: {LAMBDA_IP})")
        print(f"   2. The server might be offline - check Discord for updated IP")
        print(f"   3. Join Discord for free API access: https://algotradecamp.com")
        print(f"   4. Check the #free-api-keys channel in Discord")
        print(f"\n🌙 If running Ollama locally, use 'localhost' instead:")
        print(f"   LAMBDA_IP = 'localhost'")
        sys.exit(1)

    print(f"✅ Connected to server!")

    try:
        # Initialize the client with your local server
        client = OpenAI(
            api_key="not-needed",
            base_url=base_url,
            timeout=60.0  # Increase timeout for slower connections
        )

        print(f"\n🤖 Sending request to DeepSeek-R1...")
        # Make a chat completion request
        response = client.chat.completions.create(
            model="deepseek-r1",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": PROMPT},
            ],
            stream=False
        )

        print(f"\n🌙 MoonDev's DeepSeek Response:")
        print(f"🤖 Prompt: {PROMPT}")
        print(f"✨ Response: {response.choices[0].message.content}")

    except Exception as e:
        print(f"\n❌ Error communicating with server: {str(e)}")
        print(f"\n💡 The server might be overloaded or unavailable")
        print(f"   Try again in a few minutes or check Discord for updates")
        sys.exit(1)