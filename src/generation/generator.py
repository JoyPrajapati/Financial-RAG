import os
import re
from openai import OpenAI
from dotenv import load_dotenv
from langchain_core.documents import Document
from src.generation.prompt_template import build_prompt

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"]
)

HF_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

def generate_answer(
    question: str,
    retrieved_docs: list[Document],
    temperature: float = 0.1,
    max_new_tokens: int = 1024
) -> dict:
    
    # 1. Ensure retrieved_docs is a list (prevents NoneType iteration errors)
    if retrieved_docs is None:
        retrieved_docs = []

    system_prompt, user_prompt = build_prompt(question, retrieved_docs)

    try:
        response = client.chat.completions.create(
            model=HF_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=max_new_tokens
        )
    except Exception as e:
        raise ValueError(f"API Request Failed: {str(e)}")

    # 2. Defensive check for OpenRouter's intermittent None choices
    if not response or not hasattr(response, "choices") or not response.choices:
        raise ValueError("OpenRouter returned an empty response (choices is None). Please try again.")

    raw_answer = response.choices[0].message.content

    # 3. Defensive check in case the model returns a valid choice but empty content
    if raw_answer is None:
        raw_answer = ""

    # Strip Nvidia-nano-omni thinking block before returning
    answer = re.sub(
        r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL
    ).strip()

    # 4. Safely build sources list in case doc.metadata is None
    sources = []
    for doc in retrieved_docs:
        metadata = doc.metadata if doc.metadata is not None else {}
        sources.append({
            "section": metadata.get("section", "Unknown"),
            "source": metadata.get("source", "Unknown"),
            "text_snippet": doc.page_content[:200] if doc.page_content else ""
        })

    return {
        "answer": answer,
        "sources": sources,
        "model_used": HF_MODEL,
        "num_sources": len(retrieved_docs)
    }