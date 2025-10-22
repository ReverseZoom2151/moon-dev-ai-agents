'''
search api
computer use api
'''

# Standard library imports
import os

# Third-party from imports
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize client with API key
client = OpenAI(
    api_key=os.getenv('OPENAI_KEY'),
    base_url="https://api.openai.com/v1"  # Ensure we're using the latest API
)

try:
    # First try the new Responses API
    response = client.responses.create(
        model="gpt-4o",
        tools=[{"type": "web_search_preview"}],
        input="What was a positive news story from today?"
    )
    print("Using new Responses API:")
    print(response.output_text)
except Exception as e:
    print(f"Responses API not available yet: {str(e)}")
    
    # Fall back to chat completions
    print("\nFalling back to chat completions:")
    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that can search the web for current information."
            },
            {
                "role": "user",
                "content": "What was a positive news story from today?"
            }
        ]
    )
    print(response.choices[0].message.content)