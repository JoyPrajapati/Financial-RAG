import os
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

response = client.chat.completions.create(
    model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    messages=[
        {"role": "system", "content": "You are a financial analyst."},
        {"role": "user", "content": "What is a 10-K filing in one sentence?"}
    ],
    temperature=0.1,
    max_tokens=150
)

# --- FIX: Validate that the response contains choices ---
if not hasattr(response, "choices") or not response.choices:
    # Print the full response to help debug (optional)
    print("API returned an empty response. Full response:")
    print(response)
    raise SystemExit("No choices returned from OpenRouter. Please check your API key, model name, or try again later.")

raw = response.choices[0].message.content

# Handle the case where message.content itself is None (rare but possible)
if raw is None:
    raise SystemExit("The model returned an empty message content.")

answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
print(answer)