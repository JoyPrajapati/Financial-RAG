import os
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"]
)

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1",
    messages=[
        {"role": "system", "content": "You are a financial analyst."},
        {"role": "user", "content": "What is a 10-K filing in one sentence?"}
    ],
    temperature=0.1,
    max_tokens=150
)

raw = response.choices[0].message.content
answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
print(answer)