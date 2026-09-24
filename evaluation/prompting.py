def persona_from_metadata(metadata, fields):
    items = [f"{field}: {metadata[field]}" for field in fields if field in metadata]
    return ", ".join(items)

def build_question(entry):
    prompt = entry.get("prompt", "{}")
    subject = entry.get("subject", "")
    return prompt.replace("{}", subject)

def format_history_item(entry):
    return {
        "question": build_question(entry),
        "answer": entry.get("target", entry.get("persona_answer", "")),
        "question_id": entry.get("question_id"),
    }

def history_from_entry(entry):
    history = entry.get("history") or entry.get("retrieved_history") or entry.get("past_responses")
    if isinstance(history, str):
        return history
    if not isinstance(history, list):
        return ""

    lines = []
    for item in history:
        if isinstance(item, str):
            lines.append(item)
            continue
        if not isinstance(item, dict):
            continue
        question = item.get("question") or item.get("prompt") or item.get("past_question")
        answer = item.get("answer") or item.get("target") or item.get("past_answer")
        if question and answer:
            lines.append(f"[Past Question]: {question}\n[Your Past Answer]: {answer}")
    return "\n".join(lines)


def render_prompt(template, entry, metadata, persona_fields):
    persona = persona_from_metadata(metadata, persona_fields)
    return template.format(
        persona=persona,
        question=build_question(entry),
        subject=entry.get("subject", ""),
        history=history_from_entry(entry),
    )
