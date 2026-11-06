
import os
import time

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

print("Model:", model)
print("Testing Gemini Interactions API...")

start = time.perf_counter()

try:
    response = client.interactions.create(
        model=model,
        input="Reply with exactly: OK",
        generation_config={"thinking_level": "low"},
    )

    print("Response:", response.output_text)
except Exception as exc:
    print("Error:", type(exc).__name__, str(exc))
finally:
    print(f"Elapsed: {time.perf_counter() - start:.2f}s")
