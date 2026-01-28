def summarize_conversation(llm, old_summary, recent_messages):
    text = "\n".join(
        f"{m['role']}: {m['content']}" for m in recent_messages
    )

    prompt = f"""
Summarize the conversation concisely.

Existing summary:
{old_summary}

New messages:
{text}
"""
    return llm.invoke(prompt).content


def get_recent_messages(messages, max_turns=6):
    return messages[-max_turns:]
