import os
import re
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"]
)

HF_MODEL = "deepseek-ai/DeepSeek-R1"

MULTI_QUERY_PROMPT = """You are an AI assistant helping retrieve relevant \
passages from SEC 10-K filings.
Given the user question below, generate {n} different search queries that \
would help find relevant information.
Return ONLY the queries, one per line, no numbering, no extra text.

User question: {question}

Search queries:"""


def generate_multi_queries(question: str, n: int = 3) -> list[str]:
    prompt = MULTI_QUERY_PROMPT.format(n=n, question=question)

    response = client.chat.completions.create(
        model=HF_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200
    )

    raw = response.choices[0].message.content
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    queries = [q.strip() for q in raw.strip().split("\n") if q.strip()]
    queries = queries[:n]

    if question not in queries:
        queries.insert(0, question)

    print(f"Generated {len(queries)} queries:")
    for q in queries:
        print(f"  - {q}")

    return queries