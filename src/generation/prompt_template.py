SYSTEM_PROMPT = """You are a financial analyst assistant specializing \
in SEC 10-K filings.
Answer questions accurately based ONLY on the provided context.
If the context does not contain enough information, say so clearly.
Always cite which section (Item 1, Item 1A, Item 7, etc.) your answer \
is based on."""

USER_PROMPT_TEMPLATE = """Context from SEC 10-K filing:

{context}

---

Question: {question}

Answer:"""


def build_prompt(question: str, retrieved_docs: list) -> tuple[str, str]:
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        section = doc.metadata.get("section", "unknown")
        source = doc.metadata.get("source", "unknown")
        context_parts.append(
            f"[{i}] Source: {source} | Section: {section}\n"
            f"{doc.page_content}"
        )
    context = "\n\n".join(context_parts)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )
    return SYSTEM_PROMPT, user_prompt