import typing

import torch

from experiments.py.eval_utils_counterfact import test_generation


def compute_rewrite_quality_opinionqa(
    model,
    tok,
    record: typing.Dict,
    snips=None,
    vec=None,
) -> typing.Dict:
    def ensure_plain_str(x):
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            return x.get("str", str(x))
        return str(x)

    subject = record["requested_rewrite"]["subject"]
    target_new = ensure_plain_str(record["requested_rewrite"]["target_new"])

    rewrite_prompts = [ensure_plain_str(record["requested_rewrite"]["prompt"].format(subject))]
    paraphrase_prompts = [ensure_plain_str(p) for p in record.get("paraphrase_prompts", [])]
    generation_prompts = [ensure_plain_str(p) for p in record.get("generation_prompts", [])]

    target_tok = tok(" " + target_new)["input_ids"]
    is_llama = "llama" in model.config._name_or_path.lower()
    if is_llama:
        # Llama tokenizer dummy prefix
        target_tok = target_tok[1:]

    def test_zsre_stability(prompts):
        results = []
        for prompt in prompts:
            is_step_correct = []

            for i in range(len(target_tok)):
                current_prefix = prompt + tok.decode(target_tok[:i])

                if is_llama and i > 0:
                    current_prefix = prompt + " " + tok.decode(target_tok[:i]).strip()

                inputs = tok(current_prefix, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    logits = model(**inputs).logits[:, -1, :]
                    pred_id = torch.argmax(logits, dim=-1).item()

                is_step_correct.append(pred_id == target_tok[i])

            results.append(all(is_step_correct))
        return results

    rewrite_correct = test_zsre_stability(rewrite_prompts)
    paraphrase_correct = test_zsre_stability(paraphrase_prompts)

    all_gen_inputs = rewrite_prompts + paraphrase_prompts + generation_prompts
    ret = {}

    if all_gen_inputs and snips is not None:
        try:
            rel_id = record["requested_rewrite"].get("relation_id", "default")
            consistency_texts = [
                x["text"]
                for x in snips[rel_id][record["requested_rewrite"].get("target_new_id", 0)]
            ]
        except (KeyError, TypeError):
            consistency_texts = [target_new]

        gen_stats = test_generation(
            model,
            tok,
            all_gen_inputs,
            consistency_texts,
            [],
            vec,
        )

        all_texts = gen_stats.get("text", [])
        n_rew = len(rewrite_prompts)
        n_para = len(paraphrase_prompts)

        ret["gen_original"] = all_texts[:n_rew]
        ret["gen_paraphrase"] = all_texts[n_rew : n_rew + n_para]
        ret["gen_implicit"] = all_texts[n_rew + n_para :]

        ret.update({k: v for k, v in gen_stats.items() if k != "text"})

    ret.update(
        {
            "rewrite_prompts_correct": rewrite_correct,
            "paraphrase_prompts_correct": paraphrase_correct,
        }
    )

    return ret
