import json
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def resolve_opinionqa_data_path(data_dir: str, data_rel_path: str) -> Path:
    candidate = Path(data_rel_path)
    if candidate.is_absolute():
        return candidate

    data_root = Path(data_dir)
    candidates = [
        Path.cwd() / candidate,
        data_root / candidate,
        data_root / "edit_sets" / candidate,
        data_root / "OpinionQA" / candidate,
    ]

    for path in candidates:
        if path.exists():
            return path

    return candidates[0]


class OpinionQADataset(Dataset):
    def __init__(self, data_dir: str, tok: AutoTokenizer, size=None, use_persona=False, data_rel_path=None, *args, **kwargs):

        if data_rel_path is None:
            raise ValueError("OpinionQADataset requires data_rel_path. Pass --ds_path or --ds_folder.")

        data_path = resolve_opinionqa_data_path(data_dir, data_rel_path)

        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            raw_json = json.load(f)

        # With Persona
        persona_prefix = ""
        if use_persona:
            raw_metadata = raw_json.get("metadata", {})

            target_keys = [
                'CREGION', 'AGE', 'SEX', 'RACE', 'CITIZEN', 'MARITAL', 
                'RELIG', 'EDUCATION', 'POLPARTY', 'INCOME', 'RELIGATTEND', 'POLIDEOLOGY'
            ]

            persona_items = [
                f"{k}: {raw_metadata[k]}" 
                for k in target_keys if k in raw_metadata
            ]
            persona_str = ", ".join(persona_items)
            
            persona_prefix = (
                f"You are a survey respondent with the following profile: {persona_str}.\n"
                f"Please answer the survey question based on this profile.\n"
            )

        data = []
        entries = raw_json.get("entries", [])
        
        for i, record in enumerate(entries):
            prompt_template = record.get("prompt", "{}")
            subject = record.get("subject", "")
            target_new = record.get("target", "")

            final_prompt = f"{persona_prefix}Question: {prompt_template}\nAnswer:"
            # Paraphrase
            raw_rephrases = record.get("question_paraphrased", [])
            if isinstance(raw_rephrases, str):
                raw_rephrases = [raw_rephrases] if raw_rephrases else []
            
            if not raw_rephrases:
                raw_rephrases = [prompt_template.format(subject)]
            
            paraphrase_prompts = [
                f"{persona_prefix}Question: {p}\nAnswer:"  
                for p in raw_rephrases
            ]

            # AlphaEdit format
            data.append(
                {
                    "case_id": i,
                    "requested_rewrite": {
                        "prompt": final_prompt,
                        "subject": subject,
                        "target_new": {"str": target_new}, 
                    },
                    "paraphrase_prompts": paraphrase_prompts,
                    "neighborhood_prompts": [], 
                    "attribute_prompts": [],
                    "generation_prompts": [],
                }
            )

        self._data = data[:size]
        print(f"Loaded OpinionQA dataset from {data_path} (use_persona={use_persona}); total={len(self._data)}")

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
