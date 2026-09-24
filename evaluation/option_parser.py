import re

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def retrieval_tokens(text):
    return normalize_text(text).split()

def extract_options(entry):
    match = re.search(r"\(Options:\s*(.*?)\)\s*$", str(entry.get("prompt", "")))
    if not match:
        return []
    return [option.strip() for option in match.group(1).split("|") if option.strip()]

def parse_option_answer(generated, options):
    generated_norm = normalize_text(generated)
    if not generated_norm:
        return None

    normalized_options = [(option, normalize_text(option)) for option in options]
    for option, option_norm in normalized_options:
        if generated_norm == option_norm:
            return option

    matches = []
    for option, option_norm in normalized_options:
        if generated_norm.startswith(option_norm) or re.search(rf"\b{re.escape(option_norm)}\b", generated_norm):
            matches.append((len(option_norm), option))

    if not matches:
        return None
    
    if len(matches) > 1:
        return None
    
    return matches[0][1]

def is_correct_answer(generated, target, entry):
    options = extract_options(entry)
    parsed_answer = parse_option_answer(generated, options)
    if parsed_answer is None:
        return False
    return normalize_text(parsed_answer) == normalize_text(target)

def clean_generated_answer(generated):
    return str(generated).splitlines()[0].strip()