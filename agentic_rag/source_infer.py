def infer_source(user_input: str):
    if not user_input:
        return None, None

    text = user_input.strip()

    if text.lower() in {"pdf", "p"}:
        return "pdf", None

    if text.lower() in {"url", "u"}:
        return "url", None

    if text.lower().endswith(".pdf"):
        return "pdf", text

    if text.startswith("http://") or text.startswith("https://"):
        return "url", text

    return None, None
