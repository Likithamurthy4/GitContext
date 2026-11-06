
import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

start = time.time()

print("Testing Gemini response...")

response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents="Explain Python functions in 3 short lines."
)

print(response.text)
print(f"\nTotal time: {time.time() - start:.2f} seconds")
