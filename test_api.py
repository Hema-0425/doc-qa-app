import os
import time
from google import genai
from google.genai import types

client = genai.Client(
    api_key=os.environ["GOOGLE_API_KEY"],
    http_options=types.HttpOptions(timeout=120000),
)

print("Models available to your key:")
for m in client.models.list():
    if "generateContent" in (m.supported_actions or []):
        print(" ", m.name)

model = input("\nType a model name from the list above (without 'models/'): ").strip()
start = time.time()
try:
    r = client.models.generate_content(model=model, contents="Say hello in one word.")
    print("Reply:", r.text, f"({time.time() - start:.1f} seconds)")
except Exception as e:
    print("ERROR:", e, f"({time.time() - start:.1f} seconds)")