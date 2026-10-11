"""
LLM-KVOps: Real Offline GPU LLM Execution Engine
Runs true offline Transformer models on NVIDIA GPU / CUDA with real-time KV tensor manipulation.
"""

from typing import Generator, Dict, Any, Optional, List, Tuple, Union
import time
import os
import sys

# Python 3.12 Streamlit Threading Fix
import threading
if hasattr(threading, "_register_atexit"):
    _orig_reg = threading._register_atexit
    def _safe_register_atexit(*args, **kwargs):
        try:
            return _orig_reg(*args, **kwargs)
        except RuntimeError:
            pass
    threading._register_atexit = _safe_register_atexit

try:
    import concurrent.futures.process
    import joblib
except Exception:
    pass

import psutil
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


from src.algorithms.knapsack_dp import KnapsackTokenEvictor
from src.algorithms.rabin_karp import RabinKarpHasher, PrefixCacheTrie
from src.algorithms.sweep_line import BentleyOttmannLineSweep, TokenInterval
from src.algorithms.convex_hull import GeometricEntityMerger, GrahamsScanConvexHull, Point2D

class RealLLMRunner:
    """
    Executes actual Transformer language models on local GPU / CPU,
    comparing Standard Full KV-Cache vs LLM-KVOps Combinatorial Eviction in real time.
    """
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: Optional[str] = None,
        hf_token: Optional[str] = None
    ):
        self.model_name = model_name
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[*] Initializing RealLLMRunner with model '{model_name}' on device '{self.device}'...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=True,
                token=self.hf_token
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                token=self.hf_token
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with float16 for RTX 4050 GPU compatibility
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch_dtype,
                attn_implementation="eager",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                local_files_only=True,
                token=self.hf_token
            ).to(self.device)
        except Exception:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch_dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                    token=self.hf_token
                ).to(self.device)
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    dtype=torch_dtype,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    token=self.hf_token
                ).to(self.device)

        self.model.eval()


        # DAA Algorithmic Engines
        self.knapsack_evictor = KnapsackTokenEvictor(num_sink_tokens=4, recent_window_tokens=48)
        self.prefix_trie = PrefixCacheTrie(hasher=RabinKarpHasher())
        self.sweep_line = BentleyOttmannLineSweep()
        self.convex_hull = GrahamsScanConvexHull()
        self.process = psutil.Process(os.getpid())

        # Detect if this is an instruction-tuned model that needs chat templates
        model_lower = model_name.lower()
        self.is_instruct_model = any(tag in model_lower for tag in [
            "instruct", "chat", "-it"
        ])
        print(f"[*] Model '{model_name}' loaded. Instruct model: {self.is_instruct_model}, Device: {self.device}")

    def _prepare_input_ids(self, prompt: Union[str, List[Dict[str, str]]]) -> torch.Tensor:
        """
        Prepares input_ids with proper chat template for instruction-tuned models.
        Supports single prompt string or multi-turn list of message dicts.
        Completion models (GPT-2, DistilGPT-2) get dialogue-formatted tokenization.
        Always returns a 2D tensor of shape [1, seq_len].
        """
        if isinstance(prompt, list):
            raw_messages = []
            for m in prompt:
                r = m.get("role", "user")
                if r in ["bot", "assistant"]:
                    role = "assistant"
                elif r == "system":
                    role = "system"
                else:
                    role = "user"
                c = m.get("content") or m.get("text", "")
                if c.strip():
                    raw_messages.append({"role": role, "content": c})
        else:
            raw_messages = [
                {"role": "user", "content": str(prompt)}
            ]

        if self.is_instruct_model and hasattr(self.tokenizer, 'apply_chat_template'):
            messages = []
            has_system = any(m.get("role") == "system" for m in raw_messages)
            if not has_system:
                messages.append({"role": "system", "content": "You are a helpful assistant. Answer questions clearly and concisely."})
            messages.extend(raw_messages)

            try:
                result = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                if hasattr(result, 'input_ids'):
                    input_ids = result.input_ids
                elif isinstance(result, torch.Tensor):
                    input_ids = result
                elif isinstance(result, list):
                    input_ids = torch.tensor([result], dtype=torch.long)
                else:
                    input_ids = result

                if isinstance(input_ids, torch.Tensor):
                    if input_ids.dim() == 1:
                        input_ids = input_ids.unsqueeze(0)
                    return input_ids.to(self.device)
                else:
                    return torch.tensor([input_ids], dtype=torch.long, device=self.device)
            except Exception as e:
                print(f"[!] Chat template failed ({e}), falling back to raw tokenization")

        # Fallback for base completion models:
        if isinstance(prompt, list):
            formatted_text = ""
            for m in raw_messages:
                prefix = "User: " if m["role"] == "user" else "Assistant: "
                formatted_text += f"{prefix}{m['content']}\n"
            formatted_text += "Assistant: "
        else:
            formatted_text = str(prompt)

        return self.tokenizer(formatted_text, return_tensors="pt").input_ids.to(self.device)

    def get_hardware_memory(self) -> Dict[str, float]:
        """Returns instantaneous system RAM and GPU VRAM in Megabytes."""
        ram_mb = self.process.memory_info().rss / (1024.0 * 1024.0)
        gpu_alloc_mb = 0.0
        gpu_res_mb = 0.0
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_alloc_mb = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
            gpu_res_mb = torch.cuda.memory_reserved() / (1024.0 * 1024.0)
        return {
            "ram_mb": ram_mb,
            "gpu_allocated_mb": gpu_alloc_mb,
            "gpu_reserved_mb": gpu_res_mb
        }

    def compute_kv_cache_bytes(self, past_key_values) -> int:
        """Calculates exact total byte footprint of past_key_values across all layers."""
        if past_key_values is None:
            return 0
        total_bytes = 0
        # Transformers 5.0+ DynamicCache with .layers
        if hasattr(past_key_values, "layers"):
            for layer in past_key_values.layers:
                if hasattr(layer, "keys") and hasattr(layer, "values"):
                    total_bytes += layer.keys.numel() * layer.keys.element_size() + layer.values.numel() * layer.values.element_size()
        # Transformers 4.x DynamicCache with .key_cache
        elif hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
            for k, v in zip(past_key_values.key_cache, past_key_values.value_cache):
                total_bytes += k.numel() * k.element_size() + v.numel() * v.element_size()
        # Standard tuple/list of (key, value) tensors
        elif isinstance(past_key_values, (tuple, list)):
            for layer in past_key_values:
                if isinstance(layer, (tuple, list)) and len(layer) >= 2:
                    k, v = layer[0], layer[1]
                    total_bytes += k.numel() * k.element_size() + v.numel() * v.element_size()
        return total_bytes

    def generate_streaming_baseline(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 40,
        repetition_penalty: float = 1.15
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Runs standard offline autoregressive generation (Without Eviction).
        Yields real-time step telemetry as each token is computed on the GPU.
        """
        if self.device == "cuda":
            torch.cuda.empty_cache()

        input_ids = self._prepare_input_ids(prompt)
        curr_token = input_ids
        past_key_values = None
        generated_tokens = []
        generated_text = ""

        prompt_len = input_ids.shape[1]

        for step in range(max_new_tokens):
            t0 = time.perf_counter()
            with torch.no_grad():
                if step == 0:
                    pos_ids = None
                    c_pos = None
                else:
                    pos_ids = torch.tensor([[prompt_len + step]], device=self.device, dtype=torch.long)
                    c_pos = torch.tensor([past_key_values.get_seq_length()], device=self.device, dtype=torch.long) if hasattr(past_key_values, "get_seq_length") else None

                outputs = self.model(
                    input_ids=curr_token,
                    position_ids=pos_ids,
                    cache_position=c_pos,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=True
                )
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                # Sliding-window repetition penalty (penalize only recent 32 tokens to prevent loops without destroying global syntax)
                if repetition_penalty != 1.0 and generated_tokens:
                    recent_tokens = set(generated_tokens[-32:])
                    for token_id in recent_tokens:
                        if logits[0, token_id] < 0:
                            logits[0, token_id] = logits[0, token_id] * repetition_penalty
                        else:
                            logits[0, token_id] = logits[0, token_id] / repetition_penalty

                # Temperature & Top-K sampling
                if temperature > 0:
                    logits = logits / temperature
                    if top_k > 0:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float('Inf')
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

            step_time_ms = (time.perf_counter() - t0) * 1000.0
            generated_tokens.append(next_token.item())
            curr_token = next_token

            eos_ids = {self.tokenizer.eos_token_id}
            if getattr(self.tokenizer, "all_special_ids", None):
                for sp_id in self.tokenizer.all_special_ids:
                    try:
                        sp_tok = self.tokenizer.decode([sp_id]).strip()
                        if sp_tok in ["<|im_end|>", "<|endoftext|>", "</s>", "<eos>", "<|eot_id|>", "<|end|>"]:
                            eos_ids.add(sp_id)
                    except Exception:
                        pass
            if hasattr(self.tokenizer, "additional_special_tokens_ids") and self.tokenizer.additional_special_tokens_ids:
                eos_ids.update(self.tokenizer.additional_special_tokens_ids)

            is_eos_token = next_token.item() in eos_ids
            new_token_text = self.tokenizer.decode([next_token.item()], clean_up_tokenization_spaces=True)

            stop_markers = ["<|im_end|>", "<|endoftext|>", "<|im_start|>", "\nUser:"]
            hit_stop = False
            for marker in stop_markers:
                if marker in new_token_text or marker in (generated_text + new_token_text):
                    hit_stop = True
                    break

            if is_eos_token or hit_stop:
                for marker in stop_markers:
                    if marker in generated_text:
                        generated_text = generated_text.split(marker)[0]
                hw = self.get_hardware_memory()
                kv_bytes = self.compute_kv_cache_bytes(past_key_values)
                yield {
                    "mode": "baseline",
                    "step": step + 1,
                    "total_context_tokens": prompt_len + step + 1,
                    "token_text": "",
                    "full_text": generated_text,
                    "kv_cache_bytes": kv_bytes,
                    "kv_cache_mb": kv_bytes / (1024.0 * 1024.0),
                    "gpu_allocated_mb": hw["gpu_allocated_mb"],
                    "gpu_reserved_mb": hw["gpu_reserved_mb"],
                    "ram_mb": hw["ram_mb"],
                    "step_time_ms": step_time_ms,
                    "is_eos": True
                }
                break

            generated_text += new_token_text

            kv_bytes = self.compute_kv_cache_bytes(past_key_values)
            hw = self.get_hardware_memory()

            yield {
                "mode": "baseline",
                "step": step + 1,
                "total_context_tokens": prompt_len + step + 1,
                "token_text": new_token_text,
                "full_text": generated_text,
                "kv_cache_bytes": kv_bytes,
                "kv_cache_mb": kv_bytes / (1024.0 * 1024.0),
                "gpu_allocated_mb": hw["gpu_allocated_mb"],
                "gpu_reserved_mb": hw["gpu_reserved_mb"],
                "ram_mb": hw["ram_mb"],
                "step_time_ms": step_time_ms,
                "is_eos": False
            }

    def generate_streaming_kvops(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        max_new_tokens: int = 512,
        budget_capacity: int = 0,
        temperature: float = 0.7,
        top_k: int = 40,
        repetition_penalty: float = 1.15,
        min_sink_tokens: int = 4
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Runs offline autoregressive generation WITH LLM-KVOps Combinatorial Eviction.
        Directly prunes GPU Key-Value tensors using 0-1 Knapsack DP with Attention Sinks.
        Yields real-time step telemetry.
        """
        if self.device == "cuda":
            torch.cuda.empty_cache()

        input_ids = self._prepare_input_ids(prompt)
        curr_token = input_ids
        past_key_values = None
        generated_tokens = []
        generated_text = ""

        prompt_len = input_ids.shape[1]
        tokens_evicted_total = 0

        # Dynamic KV budget: ensure budget is at least prompt_len + 8
        if budget_capacity <= 0:
            budget_capacity = max(128, prompt_len + 48)
        elif budget_capacity < (prompt_len + 8):
            budget_capacity = prompt_len + 8
        print(f"[*] KVOps budget_capacity set to {budget_capacity} (prompt_len={prompt_len})")

        # -------------------------------------------------------------
        # DAA TIER 1: Rabin-Karp Rolling Hash & Prefix Trie Deduplication
        # -------------------------------------------------------------
        input_token_list = input_ids[0].tolist()
        prefix_match_len, cache_ref, _ = self.prefix_trie.match_longest_prefix(input_token_list)
        if prefix_match_len == 0 and len(input_token_list) >= 4:
            self.prefix_trie.insert_prefix(input_token_list[:min(len(input_token_list), 16)], cache_ref_id=f"prompt_{int(time.time())}")

        sweep_overlaps_count = 0
        hull_vertices_count = 0
        retained_indices = None

        for step in range(max_new_tokens):
            t0 = time.perf_counter()
            with torch.no_grad():
                if step == 0:
                    pos_ids = None
                    c_pos = None
                else:
                    pos_ids = torch.tensor([[prompt_len + step]], device=self.device, dtype=torch.long)
                    c_pos = torch.tensor([past_key_values.get_seq_length()], device=self.device, dtype=torch.long) if hasattr(past_key_values, "get_seq_length") else None

                outputs = self.model(
                    input_ids=curr_token,
                    position_ids=pos_ids,
                    cache_position=c_pos,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=True
                )
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                # Sliding-window repetition penalty (penalize only recent 32 tokens to prevent loops without destroying global syntax)
                if repetition_penalty != 1.0 and generated_tokens:
                    recent_tokens = set(generated_tokens[-32:])
                    for token_id in recent_tokens:
                        if logits[0, token_id] < 0:
                            logits[0, token_id] = logits[0, token_id] * repetition_penalty
                        else:
                            logits[0, token_id] = logits[0, token_id] / repetition_penalty

                # Temperature & Top-K sampling
                if temperature > 0:
                    logits = logits / temperature
                    if top_k > 0:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float('Inf')
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

            generated_tokens.append(next_token.item())
            curr_token = next_token

            eos_ids = {self.tokenizer.eos_token_id}
            if getattr(self.tokenizer, "all_special_ids", None):
                for sp_id in self.tokenizer.all_special_ids:
                    try:
                        sp_tok = self.tokenizer.decode([sp_id]).strip()
                        if sp_tok in ["<|im_end|>", "<|endoftext|>", "</s>", "<eos>", "<|eot_id|>", "<|end|>"]:
                            eos_ids.add(sp_id)
                    except Exception:
                        pass
            if hasattr(self.tokenizer, "additional_special_tokens_ids") and self.tokenizer.additional_special_tokens_ids:
                eos_ids.update(self.tokenizer.additional_special_tokens_ids)

            is_eos_token = next_token.item() in eos_ids
            new_token_text = self.tokenizer.decode([next_token.item()], clean_up_tokenization_spaces=True)

            stop_markers = ["<|im_end|>", "<|endoftext|>", "<|im_start|>", "\nUser:"]
            hit_stop = False
            for marker in stop_markers:
                if marker in new_token_text or marker in (generated_text + new_token_text):
                    hit_stop = True
                    break

            if is_eos_token or hit_stop:
                for marker in stop_markers:
                    if marker in generated_text:
                        generated_text = generated_text.split(marker)[0]
                step_time_ms = (time.perf_counter() - t0) * 1000.0
                kv_bytes = self.compute_kv_cache_bytes(past_key_values)
                hw = self.get_hardware_memory()
                yield {
                    "mode": "kvops",
                    "step": step + 1,
                    "total_context_tokens": prompt_len + step + 1,
                    "retained_slots": len(retained_indices) if retained_indices is not None else (prompt_len + step + 1),
                    "tokens_evicted": tokens_evicted_total,
                    "token_text": "",
                    "full_text": generated_text,
                    "kv_cache_bytes": kv_bytes,
                    "kv_cache_mb": kv_bytes / (1024.0 * 1024.0),
                    "gpu_allocated_mb": hw["gpu_allocated_mb"],
                    "gpu_reserved_mb": hw["gpu_reserved_mb"],
                    "ram_mb": hw["ram_mb"],
                    "step_time_ms": step_time_ms,
                    "retained_indices": retained_indices,
                    "daa_telemetry": {
                        "rabin_karp_prefix_match": prefix_match_len,
                        "sweep_line_overlaps": sweep_overlaps_count,
                        "convex_hull_vertices": hull_vertices_count,
                        "knapsack_retained": len(retained_indices) if retained_indices is not None else (prompt_len + step + 1),
                        "knapsack_evicted": tokens_evicted_total,
                        "attention_sinks": self.knapsack_evictor.num_sink_tokens
                    },
                    "is_eos": True
                }
                break

            generated_text += new_token_text

            # -------------------------------------------------------------
            # DAA TIER 2 & 3: Line Sweep + Convex Hull + 0-1 Knapsack DP
            # -------------------------------------------------------------
            retained_indices = None
            if past_key_values is not None:
                # Determine current sequence length in KV tensors
                if hasattr(past_key_values, "get_seq_length"):
                    curr_seq_len = past_key_values.get_seq_length()
                elif hasattr(past_key_values, "layers") and len(past_key_values.layers) > 0:
                    curr_seq_len = past_key_values.layers[0].keys.shape[-2]
                elif hasattr(past_key_values, "key_cache") and len(past_key_values.key_cache) > 0:
                    curr_seq_len = past_key_values.key_cache[0].shape[-2]
                elif isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
                    curr_seq_len = past_key_values[0][0].shape[-2]
                else:
                    curr_seq_len = 0

                if curr_seq_len > budget_capacity:
                    # Extract last-layer attention distribution
                    if outputs.attentions is not None and len(outputs.attentions) > 0:
                        last_att = outputs.attentions[-1][0].mean(dim=0)[-1].detach().cpu().numpy()
                        if len(last_att) != curr_seq_len:
                            if len(last_att) > curr_seq_len:
                                last_att = last_att[:curr_seq_len]
                            else:
                                pad = np.ones(curr_seq_len - len(last_att), dtype=np.float32) * float(np.mean(last_att))
                                last_att = np.concatenate([last_att, pad])
                    else:
                        last_att = np.ones(curr_seq_len, dtype=np.float32)

                    # Tier 2A: Bentley-Ottmann Line Sweep (Segment Intersection)
                    intervals = [
                        TokenInterval(i, float(i), float(i) + float(last_att[i] * 10.0), layer_id=0)
                        for i in range(min(len(last_att), 32))
                    ]
                    sweep_overlaps_count = len(self.sweep_line.find_intersections(intervals))

                    # Tier 2B: Graham's Scan Convex Hull
                    points_2d = [
                        Point2D(x=float(i), y=float(last_att[i] * 100.0), token_id=i)
                        for i in range(min(len(last_att), 32))
                    ]
                    hull_vertices = self.convex_hull.compute_hull(points_2d)
                    hull_vertices_count = len(hull_vertices)

                    # Tier 3: 0-1 Knapsack Dynamic Programming with Attention Sinks
                    token_positions = np.arange(curr_seq_len)
                    utility_scores = self.knapsack_evictor.compute_utility_scores(
                        attention_weights=last_att,
                        token_positions=token_positions,
                        current_time_step=curr_seq_len
                    )

                    # Preserve prompt tokens as sinks to prevent prompt corruption
                    self.knapsack_evictor.num_sink_tokens = max(4, prompt_len)
                    retained_indices, _, _ = self.knapsack_evictor.solve_01_knapsack(
                        utility_scores=utility_scores,
                        token_weights=None,
                        budget_capacity=budget_capacity
                    )

                    # Slice the actual PyTorch GPU Tensors along the sequence dimension
                    retained_tensor_idx = torch.tensor(retained_indices, device=self.device, dtype=torch.long)
                    
                    if hasattr(past_key_values, "layers"):
                        for layer in past_key_values.layers:
                            if hasattr(layer, "keys") and hasattr(layer, "values"):
                                layer.keys = layer.keys.index_select(-2, retained_tensor_idx)
                                layer.values = layer.values.index_select(-2, retained_tensor_idx)
                    elif hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
                        for i in range(len(past_key_values.key_cache)):
                            past_key_values.key_cache[i] = past_key_values.key_cache[i].index_select(-2, retained_tensor_idx)
                            past_key_values.value_cache[i] = past_key_values.value_cache[i].index_select(-2, retained_tensor_idx)
                    elif isinstance(past_key_values, (tuple, list)):
                        new_layers = []
                        for k, v in past_key_values:
                            new_k = k.index_select(-2, retained_tensor_idx)
                            new_v = v.index_select(-2, retained_tensor_idx)
                            new_layers.append((new_k, new_v))
                        past_key_values = tuple(new_layers)

                    tokens_evicted_total = (prompt_len + step + 1) - len(retained_indices)

            step_time_ms = (time.perf_counter() - t0) * 1000.0
            kv_bytes = self.compute_kv_cache_bytes(past_key_values)
            hw = self.get_hardware_memory()

            yield {
                "mode": "kvops",
                "step": step + 1,
                "total_context_tokens": prompt_len + step + 1,
                "retained_slots": len(retained_indices) if retained_indices is not None else (prompt_len + step + 1),
                "tokens_evicted": tokens_evicted_total,
                "token_text": new_token_text,
                "full_text": generated_text,
                "kv_cache_bytes": kv_bytes,
                "kv_cache_mb": kv_bytes / (1024.0 * 1024.0),
                "gpu_allocated_mb": hw["gpu_allocated_mb"],
                "gpu_reserved_mb": hw["gpu_reserved_mb"],
                "ram_mb": hw["ram_mb"],
                "step_time_ms": step_time_ms,
                "retained_indices": retained_indices,
                "daa_telemetry": {
                    "rabin_karp_prefix_match": prefix_match_len,
                    "sweep_line_overlaps": sweep_overlaps_count,
                    "convex_hull_vertices": hull_vertices_count,
                    "knapsack_retained": len(retained_indices) if retained_indices is not None else (prompt_len + step + 1),
                    "knapsack_evicted": tokens_evicted_total,
                    "attention_sinks": self.knapsack_evictor.num_sink_tokens
                },
                "is_eos": False
            }
