import torch
from pathlib import Path

from utils import extract_respondent_id, extract_config_name

def apply_edited_weights(model, weights_path):
    edited_weights = torch.load(weights_path, map_location="cpu")
    model_state = model.state_dict()
    original_weights = {}
    missing = []

    with torch.no_grad():
        for name, value in edited_weights.items():
            if name not in model_state:
                missing.append(name)
                continue
            original_weights[name] = model_state[name].detach().cpu().clone()
            model_state[name].copy_(value.to(device=model_state[name].device, dtype=model_state[name].dtype))

    if missing:
        print(f"Skipped {len(missing)} tensors not present in the model.")

    return original_weights


def reset_edited_weights(model, original_weights):
    if not original_weights:
        return

    model_state = model.state_dict()
    with torch.no_grad():
        for name, value in original_weights.items():
            if name in model_state:
                model_state[name].copy_(value.to(device=model_state[name].device, dtype=model_state[name].dtype))

def discover_weight_files(weights_root: Path):
    candidates = list(weights_root.rglob("model_delta.pt"))
    candidates.extend(weights_root.rglob("*_model_delta.pt"))
    return sorted(set(candidates))

def build_weight_index(weights_root: Path):
    indexed = []
    for weight_path in discover_weight_files(weights_root):
        indexed.append(
            {
                "path": weight_path,
                "respondent_id": extract_respondent_id(weight_path),
                "config_name": extract_config_name(weight_path),
                "text": f"{weight_path.parent.name}/{weight_path.name}",
            }
        )
    return indexed

def match_weight_file(input_path: Path, weight_index):
    respondent_id = extract_respondent_id(input_path)
    config_name = extract_config_name(input_path)
    matches = []

    for item in weight_index:
        if respondent_id and item["respondent_id"] != respondent_id:
            continue
        score = 1
        if config_name and item["config_name"] == config_name:
            score += 1
        matches.append((score, item["path"]))

    if not matches:
        return None

    matches.sort(key=lambda item: (-item[0], str(item[1])))
    return matches[0][1]
