import math
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from option_parser import retrieval_tokens
from prompting import build_question, format_history_item
from utils import extract_respondent_id, extract_config_name, load_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
class BM25HistoryRetriever:
    def __init__(self, entries):
        self.entries = entries
        self.documents = [retrieval_tokens(build_question(entry)) for entry in entries]
        self.avgdl = sum(len(doc) for doc in self.documents) / len(self.documents) if self.documents else 0.0
        doc_freqs = Counter()
        for doc in self.documents:
            doc_freqs.update(set(doc))
        self.idf = {
            term: math.log(1 + (len(self.documents) - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freqs.items()
        }

    def score(self, query_tokens, doc_tokens):
        if not doc_tokens:
            return 0.0
        counts = Counter(doc_tokens)
        score = 0.0
        k1 = 1.5
        b = 0.75
        for term in query_tokens:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            denom = tf + k1 * (1 - b + b * len(doc_tokens) / (self.avgdl or 1.0))
            score += self.idf.get(term, 0.0) * tf * (k1 + 1) / denom
        return score

    def retrieve(self, entry, top_k):
        query_tokens = retrieval_tokens(build_question(entry))
        query_id = entry.get("question_id")
        scored = []
        for idx, doc_tokens in enumerate(self.documents):
            candidate = self.entries[idx]
            if query_id and candidate.get("question_id") == query_id:
                continue
            scored.append((self.score(query_tokens, doc_tokens), idx))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [format_history_item(self.entries[idx]) for _, idx in scored[:top_k]]


class ContrieverHistoryRetriever:
    def __init__(self, entries, model_name, device):
        self.entries = entries
        self.texts = [build_question(entry) for entry in entries]
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.embeddings = self.embed(self.texts)

    def embed(self, texts):
        all_embeddings = []
        for start in range(0, len(texts), 32):
            batch = texts[start : start + 32]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, dim=0) if all_embeddings else torch.empty(0, device=self.device)

    def retrieve(self, entry, top_k):
        if len(self.entries) == 0:
            return []
        query = self.embed([build_question(entry)])
        similarities = torch.mm(query, self.embeddings.t()).squeeze(0)
        query_id = entry.get("question_id")
        if query_id:
            for idx, candidate in enumerate(self.entries):
                if candidate.get("question_id") == query_id:
                    similarities[idx] = -float("inf")
        k = min(top_k, len(self.entries))
        top_indices = torch.topk(similarities, k=k).indices.cpu().tolist()
        return [
            format_history_item(self.entries[idx])
            for idx in top_indices
            if torch.isfinite(similarities[idx]).item()
        ]


def build_history_retriever(input_path, args, contriever_cache):
    if args.history_mode == "none":
        return None, None

    history_file = find_history_file(input_path, args)
    if history_file is None:
        print(f"No history file found for {input_path}; prompts will use empty history.")
        return None, None

    payload = load_json(history_file)
    entries = payload.get("entries", [])
    if args.history_mode == "bm25":
        return BM25HistoryRetriever(entries), history_file

    cache_key = str(history_file.resolve())
    if cache_key not in contriever_cache:
        device = args.history_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        contriever_cache[cache_key] = ContrieverHistoryRetriever(entries, args.contriever_model, device)
    return contriever_cache[cache_key], history_file

def find_history_file(input_path, args):
    respondent_id = extract_respondent_id(input_path)
    config_name = extract_config_name(input_path) or args.eval_config
    if respondent_id is None:
        return None

    search_dirs = []
    if args.history_dir is not None:
        search_dirs.append(args.history_dir)
    if config_name:
        search_dirs.append(PROJECT_ROOT / "data/edit_sets" / config_name)

    for directory in search_dirs:
        if directory is None or not directory.exists():
            continue
        matches = sorted(directory.glob(f"*{respondent_id}*.json"))
        if matches:
            return matches[0]
    return None

def attach_retrieved_history(entries, retriever, args):
    if retriever is None:
        return entries

    augmented = []
    for entry in entries:
        copied = dict(entry)
        copied["retrieved_history"] = retriever.retrieve(entry, args.history_top_k)
        augmented.append(copied)
    return augmented
