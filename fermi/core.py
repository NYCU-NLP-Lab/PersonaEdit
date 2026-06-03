import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class FERMI:
    def __init__(
        self,
        pred_model,
        opt_model,
        user_profile,
        user_opinions,
        fixed_demos,
        embedding_device="cpu",
        embedding_batch_size=32,
        score_batch_size=4,
        subset_batch_size=1,
        memory_failure_case_limit=None,
        precomputed_opi_embs=None,
        precomputed_test_embs=None,
        user_id=None,
        num_generate_prompts=4,
    ):
        self.user_id = user_id or "unknown"
        self.M = pred_model
        self.M_opt = opt_model
        self.U_pro = user_profile
        self.U_opi = user_opinions
        self.memory = []
        self.prompt_pool = []
        self.fixed_demos = fixed_demos
        self.embedding_batch_size = max(1, embedding_batch_size)
        self.score_batch_size = max(1, score_batch_size)
        self.subset_batch_size = max(1, subset_batch_size)
        self.memory_failure_case_limit = memory_failure_case_limit
        self.precomputed_test_embs = precomputed_test_embs or {}
        self.num_generate_prompts = max(1, num_generate_prompts)
        self.opi_texts = [item[0] for item in self.U_opi]
        self.opi_true_answers = [item[1] for item in self.U_opi]
        self.prompt_prediction_cache = None

        if precomputed_opi_embs is not None:
            self.embedding_model = None
            self.opi_embs = precomputed_opi_embs
            print("Loaded precomputed embeddings.")
        else:
            self.embedding_model = SentenceTransformer('all-mpnet-base-v2', device=embedding_device)

            print("Precomputing opinion embeddings...")
            self.opi_embs = self.embedding_model.encode(
                self.opi_texts,
                batch_size=self.embedding_batch_size,
                show_progress_bar=False,
            )

    def score_prompts(self, prompts):
        """Score prompts and collect misaligned responses."""
        scored_results = []
        questions = [item[0] for item in self.U_opi]
        true_answers = [item[1] for item in self.U_opi]
        for prompt_index, p in enumerate(prompts, start=1):
            progress_label = f"User {self.user_id} | Scoring prompt {prompt_index}/{len(prompts)}"
            print(f"[{progress_label}] start on {len(questions)} opinions", flush=True)
            pred_answers = self.M.generate_batch(
                questions,
                p,
                batch_size=self.score_batch_size,
                progress_label=progress_label,
            )
            
            correct_count = 0
            mis_aligned = []
            for i, (pred_a, true_a) in enumerate(zip(pred_answers, true_answers)):
                if pred_a == true_a:
                    correct_count += 1
                else:
                    mis_aligned.append({
                        "idx": i, "q": questions[i], "a": true_a, "pred": pred_a
                    })
            
            score = correct_count / len(self.U_opi)
            print(f"[{progress_label}] score={score:.4f}", flush=True)
            scored_results.append({"prompt": p, "score": score, "context": mis_aligned})
        return scored_results
    
    def evaluate_on_subset(self, prompt, subset):
        """Score a prompt on a retrieval subset."""
        qs = [item[0] for item in subset]
        ans = [item[1] for item in subset]
        preds = self.M.generate_batch(qs, prompt, batch_size=self.subset_batch_size)
        
        correct = sum(1 for p, a in zip(preds, ans) if p == a)
        return correct / len(subset)

    def build_prompt_prediction_cache(self):
        """Precompute predictions for each final prompt on all U_opi questions."""
        if not self.prompt_pool:
            self.prompt_prediction_cache = None
            return

        questions = [item[0] for item in self.U_opi]
        cache = []
        print(
            f"[User {self.user_id} | PromptCache] precomputing {len(self.prompt_pool)} prompts "
            f"over {len(questions)} opinions",
            flush=True,
        )
        for prompt_index, prompt in enumerate(self.prompt_pool, start=1):
            progress_label = (
                f"User {self.user_id} | PromptCache prompt {prompt_index}/{len(self.prompt_pool)}"
            )
            preds = self.M.generate_batch(
                questions,
                prompt,
                batch_size=self.score_batch_size,
                progress_label=progress_label,
            )
            cache.append(np.array(preds, dtype=object))

        self.prompt_prediction_cache = cache
        print(
            f"[User {self.user_id} | PromptCache] ready with {len(self.prompt_prediction_cache)} prompts",
            flush=True,
        )

    def evaluate_prompt_on_indices_cached(self, prompt_index, indices):
        """Use cached predictions to score one prompt on a subset of U_opi indices."""
        if self.prompt_prediction_cache is None:
            raise ValueError("Prompt prediction cache is not built.")

        preds = self.prompt_prediction_cache[prompt_index]
        correct = sum(
            1
            for idx in indices
            if preds[idx] == self.opi_true_answers[idx]
        )
        return correct / len(indices)

    def update_memory(self, scored_results, L=5):
        """Update memory and build the optimizer context."""
        print(f"[User {self.user_id} | Memory] updating Top-{L} memory", flush=True)
        combined = self.memory + scored_results
        self.memory = sorted(combined, key=lambda x: x['score'])[-L:] 

        worst_p = self.memory[0]
        worst_error_indices = set(case['idx'] for case in worst_p['context'])
        
        formatted_context = ""
        for i, item in enumerate(self.memory):
            formatted_context += f"\ntext: Prompt #{i+1}\nscore: {item['score']:.4f}\nfailure cases: "
            
            if i == 0:
                formatted_context += "\n"
                if self.memory_failure_case_limit is None:
                    shown_cases = item['context']
                else:
                    shown_cases = item['context'][:self.memory_failure_case_limit]
                for case in shown_cases:
                    formatted_context += (
                        f"<{case['idx']}>\n"
                        f"Question: {case['q']}\n"
                        f"Answer: {case['a']}\n"
                        f"Your response: {case['pred']}\n"
                    )
                omitted_count = len(item['context']) - len(shown_cases)
                if omitted_count > 0:
                    formatted_context += f"... {omitted_count} additional failure cases omitted to fit memory.\n"
            else:
                curr_error_indices = set(case['idx'] for case in item['context'])
                common = sorted(list(worst_error_indices.intersection(curr_error_indices)))
                new_errors = len(curr_error_indices - worst_error_indices)
                formatted_context += f"{common} and {new_errors} additional examples that was correctly predicted with the first text\n"
                
        return formatted_context

    def generate_new_prompts(self, context, K=None):
        """Generate K new prompts with the optimizer model."""
        K = K or self.num_generate_prompts
        print(f"[User {self.user_id} | Optimize] Generating {K} new prompts...", flush=True)
        new_prompts = self.M_opt.optimize(context,self.fixed_demos, num_generate=K)
        print(f"[User {self.user_id} | Optimize] Generated {len(new_prompts)} new prompts.", flush=True)
        return new_prompts

    def run_optimization(self, T=10):
        """Run T optimization iterations."""
        initial_p = self.U_pro  
        current_prompts = [initial_p]

        for t in range(T):
            print(f"[User {self.user_id} | Iteration {t + 1}/{T}] Step 1 scoring", flush=True)
            results = self.score_prompts(current_prompts)
            print(f"[User {self.user_id} | Iteration {t + 1}/{T}] Step 2 update memory", flush=True)
            context = self.update_memory(results)
            print(f"[User {self.user_id} | Iteration {t + 1}/{T}] Step 3 generate prompts", flush=True)
            current_prompts = self.generate_new_prompts(context)

            if t == T - 1:
                self.prompt_pool = current_prompts

    def inference_rop(self, test_q, N_tilde=3, progress_label=None):
        """Run Retrieval-of-Prompt inference."""
        label = progress_label or f"User {self.user_id} | Inference"
        print(f"[{label}] retrieving top-{N_tilde} opinions", flush=True)

        if test_q in self.precomputed_test_embs:
            q_emb = self.precomputed_test_embs[test_q].reshape(1, -1)
        elif self.embedding_model is not None:
            q_emb = self.embedding_model.encode([test_q], show_progress_bar=False)
        else:
            raise ValueError(f"Missing precomputed embedding for test question: {test_q[:80]}")
        
        similarities = cosine_similarity(q_emb, self.opi_embs)[0]
        top_indices = np.argsort(similarities)[-N_tilde:]
        relevant_opi = [self.U_opi[i] for i in top_indices]
        
        best_p = None
        best_subset_score = -1
        if self.prompt_prediction_cache is not None:
            for prompt_index, p in enumerate(self.prompt_pool):
                print(
                    f"[{label}] scoring candidate prompt {prompt_index + 1}/{len(self.prompt_pool)} from cache",
                    flush=True,
                )
                score = self.evaluate_prompt_on_indices_cached(prompt_index, top_indices)
                if score > best_subset_score:
                    best_subset_score = score
                    best_p = p
        else:
            relevant_opi = [self.U_opi[i] for i in top_indices]
            for prompt_index, p in enumerate(self.prompt_pool, start=1):
                print(f"[{label}] scoring candidate prompt {prompt_index}/{len(self.prompt_pool)}", flush=True)
                score = self.evaluate_on_subset(p, relevant_opi)
                if score > best_subset_score:
                    best_subset_score = score
                    best_p = p
        
        print(f"[{label}] best subset score={best_subset_score:.4f}; generating final answer", flush=True)
        return self.M.generate_batch(
            [test_q],
            best_p,
            batch_size=1,
            progress_label=f"{label} final answer",
        )[0]
