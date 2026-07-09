import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.documents import Document
from src.generation.prompt_template import build_prompt

load_dotenv()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"]
)

HF_MODEL = "deepseek-ai/DeepSeek-R1"


def generate_answer(
    question: str,
    retrieved_docs: list[Document],
    temperature: float = 0.1,
    max_new_tokens: int = 1024
) -> dict:
    system_prompt, user_prompt = build_prompt(question, retrieved_docs)

    response = client.chat.completions.create(
        model=HF_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=max_new_tokens
    )

    raw_answer = response.choices[0].message.content

    # Strip DeepSeek-R1 thinking block before returning
    answer = re.sub(
        r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL
    ).strip()

    return {
        "answer": answer,
        "sources": [
            {
                "section": doc.metadata.get("section"),
                "source": doc.metadata.get("source"),
                "text_snippet": doc.page_content[:200]
            }
            for doc in retrieved_docs
        ],
        "model_used": HF_MODEL,
        "num_sources": len(retrieved_docs)
    }