import gc
import os
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

def _parse_max_memory(max_memory_text):
    if not max_memory_text:
        return None

    max_memory = {}
    for item in max_memory_text.split(","):
        key, value = item.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key.isdigit():
            key = int(key)
        max_memory[key] = value
    return max_memory


class StopOnSubstrings(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings):
        self.stop_ids = [
            tokenizer.encode(stop_string, add_special_tokens=False)
            for stop_string in stop_strings
        ]

    def __call__(self, input_ids, scores, **kwargs):
        for row in input_ids.tolist():
            if not any(
                len(row) >= len(stop_ids) and row[-len(stop_ids):] == stop_ids
                for stop_ids in self.stop_ids
                if stop_ids
            ):
                return False
        return True

class LlamaLocal:
    def __init__(
        self,
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        generation_batch_size=4,
        optimize_batch_size=1,
        max_memory=None,
        offload_folder="offload",
        use_cache=False,
        optimize_max_input_tokens=None,
        device_map="auto",
        pred_max_new_tokens=20,
        optimize_max_new_tokens=256,
        attention_implementation=None,
        matmul_precision=None,
    ):
        self.generation_batch_size = max(1, generation_batch_size)
        self.optimize_batch_size = max(1, optimize_batch_size)
        self.use_cache = use_cache
        self.optimize_max_input_tokens = optimize_max_input_tokens
        self.pred_max_new_tokens = max(1, pred_max_new_tokens)
        self.optimize_max_new_tokens = max(1, optimize_max_new_tokens)
        self.attention_implementation = attention_implementation or os.getenv("FERMI_ATTENTION_IMPL")
        self.matmul_precision = matmul_precision or os.getenv("FERMI_MATMUL_PRECISION")

        if self.matmul_precision:
            torch.set_float32_matmul_precision(self.matmul_precision)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.padding_side = "left" 
        self.tokenizer.pad_token = self.tokenizer.eos_token

        max_memory = max_memory or _parse_max_memory(os.getenv("FERMI_MAX_MEMORY"))
        if max_memory:
            os.makedirs(offload_folder, exist_ok=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map=device_map,
            attn_implementation=self._resolve_attn_impl(),
            max_memory=max_memory,
            offload_folder=offload_folder,
            offload_state_dict=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self._log_device_map()
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _resolve_attn_impl(self):
        if not self.attention_implementation:
            return None
        value = self.attention_implementation.strip().lower()
        if value in {"none", "default", "auto"}:
            return None
        if value in {"flash", "flash_attention_2", "flash-attention-2"}:
            return "flash_attention_2"
        if value in {"sdpa", "scaled_dot_product_attention"}:
            return "sdpa"
        return self.attention_implementation

    def _cleanup_cuda(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _log_device_map(self):
        hf_device_map = getattr(self.model, "hf_device_map", None)
        if not hf_device_map:
            print("[Device map] unavailable", flush=True)
            return

        counts = {}
        for module_name, device in hf_device_map.items():
            counts[str(device)] = counts.get(str(device), 0) + 1
        summary = ", ".join(f"{device}: {count} modules" for device, count in sorted(counts.items()))
        print(f"[Device map] {summary}", flush=True)
        for module_name, device in hf_device_map.items():
            print(f"[Device map] {module_name} -> {device}", flush=True)

    def _log_cuda_memory(self, label):
        if not torch.cuda.is_available():
            return
        parts = []
        for device_index in range(torch.cuda.device_count()):
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
            allocated = torch.cuda.memory_allocated(device_index)
            reserved = torch.cuda.memory_reserved(device_index)
            parts.append(
                f"cuda:{device_index} "
                f"free={free_bytes / 1024**3:.2f}GiB "
                f"total={total_bytes / 1024**3:.2f}GiB "
                f"allocated={allocated / 1024**3:.2f}GiB "
                f"reserved={reserved / 1024**3:.2f}GiB"
            )
        print(f"[CUDA memory | {label}] " + " | ".join(parts), flush=True)

    def _extract_prompt_candidate(self, content):
        final_match = re.search(r"FINAL:\s*(.+?)(?:\s+END\b|$)", content, re.IGNORECASE | re.DOTALL)
        if final_match:
            return self._clean_prompt_candidate(final_match.group(1))

        tag_candidates = re.findall(r"<NEW_TEXT>\s*(.*?)\s*</NEW_TEXT>", content, re.DOTALL | re.IGNORECASE)
        bracket_candidates = re.findall(r"\[(.*?)\]", content, re.DOTALL)
        candidates = tag_candidates + bracket_candidates

        cleaned = [text for text in (self._clean_prompt_candidate(c) for c in candidates) if text]
        if cleaned:
            cleaned.sort(key=len, reverse=True)
            return cleaned[0]

        return self._clean_prompt_candidate(content)

    def _clean_prompt_candidate(self, text):
        text = text.strip()
        text = re.sub(r"^```(?:text)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = re.sub(r"</?NEW_TEXT>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+END\s*$", "", text, flags=re.IGNORECASE)
        text = text.splitlines()[0].strip()
        text = re.sub(r"^(?:new text|text|prompt)\s*:\s*", "", text, flags=re.IGNORECASE)
        text = text.strip().strip("\"'")

        if not text:
            return None
        if text.isdigit():
            return None

        normalized = re.sub(r"\s+", " ", text).strip().lower()
        invalid_values = {
            "<ins>",
            "<text with score 1.0>",
            "text",
            "prompt",
            "insert new text here",
            "your new prompt here",
            "new prompt here",
        }
        if normalized in invalid_values:
            return None
        if len(text) < 40:
            return None

        return text

    def _extract_options(self, question_text):
        """
        Extract answer options from the `(Options: ...)` portion of a question.
        """
        match = re.search(r'\(Options:\s*(.*?)\)', question_text)
        if match:
            options_str = match.group(1)
            options = [opt.strip() for opt in options_str.split(',')]
            return sorted(options, key=len, reverse=True)
        return []
    
    @torch.no_grad()
    def generate_batch(self, questions, prompt, batch_size=16, progress_label=None):
        """
        Generate answers in batches with plain-text prompt concatenation.
        """
        all_results = []
        effective_batch_size = min(batch_size, self.generation_batch_size)
        
        for i in range(0, len(questions), effective_batch_size):
            batch_qs = questions[i : i + effective_batch_size]
            if progress_label:
                end_index = min(i + effective_batch_size, len(questions))
                print(
                    f"[{progress_label}] generating {i + 1}-{end_index}/{len(questions)}",
                    flush=True,
                )
            
            texts = [f"{prompt}\n\nQuestion: {q}\nAnswer:" for q in batch_qs]
                        
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.model.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.pred_max_new_tokens,
                do_sample=False,
                temperature=None,
                use_cache=self.use_cache,
                pad_token_id=self.tokenizer.eos_token_id
            )

            input_len = inputs.input_ids.shape[1]
            generated_tokens = outputs[:, input_len:]
            batch_responses = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            for res, q_text in zip(batch_responses, batch_qs):
                res_clean = res.strip()
                options_list = self._extract_options(q_text)
                
                matched = False
                for opt in options_list:
                    if res_clean.startswith(opt) or opt in res_clean:
                        all_results.append(opt)
                        matched = True
                        break
                
                if not matched:
                    all_results.append(res_clean)

            del inputs, outputs, generated_tokens, batch_responses
            self._cleanup_cuda()
            
        return all_results
    
    def optimize(self, context, fixed_demos, num_generate=4):
        """Generate new prompts with a plain-text meta-prompt."""

        instruction_header = (
            "I want to find the text that could make you a personalized answer for the user, "
            "based on the given personal information. "
            "If your response is identical to the person's, then you get a score of 1; "
            "otherwise, you get a score of 0.\n"
            "Here, I have some previous texts along with their corresponding average scores "
            "with 160 questions and specific cases that you failed to correctly answer. "
            "The texts are arranged in ascending order based on their scores, "
            "where higher scores indicate better quality."
        )
        application_instruction = (
            "\nThe following exemplars show how to apply your text: "
            "you replace <INS> in each input with your text, then read the input and give an output. "
            "We say your output is wrong if your output is different from the given output, "
            "and we say your output is correct if they are the same."
        )
        
        demo_text = "\n### Few-shot Demonstration (fixed) ###\n"
        for i, (q, a) in enumerate(fixed_demos):
            demo_text += f"[{i+1}]\n<INS>\nQuestion: {q}\nAnswer: {a}\n"

        task_footer = (
            "\nWrite exactly one complete new instruction text that is different from the old ones "
            "and has a score as high as possible. The instruction text must tell the answerer "
            "how to use the demographic profile and answer choices. "
            "It must be an instruction for answering future survey questions, not a survey question, "
            "not answer choices, and not a description of the output format. "
            "Do not write placeholders, numbering, examples, XML tags, or explanations. "
            "Output exactly one line in this format: FINAL: <your complete instruction text> END. "
            "Keep it under 80 words."
        )

        full_flat_prompt = (
            f"{instruction_header}\n\n"
            f"### Optimization Memory (varied) ###\n{context}\n"
            f"{application_instruction}\n"
            f"{demo_text}\n"
            f"{task_footer}"
        )

        new_prompts = []
        attempts = 0
        max_attempts = max(num_generate * 3, int(os.getenv("FERMI_OPTIMIZE_MAX_ATTEMPTS", num_generate * 3)))
        while len(new_prompts) < num_generate and attempts < max_attempts:
            current_batch_size = min(self.optimize_batch_size, num_generate - len(new_prompts))
            batch_texts = [full_flat_prompt] * current_batch_size

            tokenize_kwargs = {
                "padding": True,
                "return_tensors": "pt",
            }
            if self.optimize_max_input_tokens is not None:
                tokenize_kwargs["truncation"] = True
                tokenize_kwargs["max_length"] = self.optimize_max_input_tokens

            inputs = self.tokenizer(batch_texts, **tokenize_kwargs).to(self.model.device)

            print(
                "Sending plain-text meta-prompt batch "
                f"(size: {current_batch_size}, input tokens: {inputs.input_ids.shape[1]})..."
                ,
                flush=True,
            )
            self._log_cuda_memory("before optimize generate")

            with torch.no_grad():
                try:
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.optimize_max_new_tokens, 
                        do_sample=True,
                        temperature=1,   
                        top_p=0.95,
                        use_cache=self.use_cache,
                        pad_token_id=self.tokenizer.eos_token_id,
                        stopping_criteria=StoppingCriteriaList([
                            StopOnSubstrings(self.tokenizer, [" END", "\nEND", "END"])
                        ]),
                    )
                except torch.OutOfMemoryError:
                    self._log_cuda_memory("optimize OOM")
                    print(
                        "CUDA OOM during optimize generation. "
                        "This is usually caused by a long meta-prompt prefill plus "
                        "too little free memory left after model placement.",
                        flush=True,
                    )
                    raise
                
            input_len = inputs.input_ids.shape[1]
            generated_tokens = outputs[:, input_len:]
            batch_responses = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            
            for content in batch_responses:
                attempts += 1
                extracted_p = self._extract_prompt_candidate(content)
                if extracted_p:
                    new_prompts.append(extracted_p)
                    print(f"  [Optimize attempt {attempts}] extracted: {extracted_p[:80]}...")
                else:
                    preview = re.sub(r"\s+", " ", content.strip())[:120]
                    print(
                        f"  [Optimize attempt {attempts}] invalid output, retrying: {preview}",
                        flush=True,
                    )

                if len(new_prompts) >= num_generate or attempts >= max_attempts:
                    break

            del inputs, outputs, generated_tokens, batch_responses
            self._cleanup_cuda()

        if len(new_prompts) < num_generate:
            raise RuntimeError(
                f"Only generated {len(new_prompts)}/{num_generate} valid optimization prompts "
                f"after {attempts} attempts. Increase FERMI_OPTIMIZE_MAX_ATTEMPTS or inspect "
                "the optimize model output."
            )

        return new_prompts
