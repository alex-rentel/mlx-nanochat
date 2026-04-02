"""
Inference engine with KV cache for efficient generation.
Note: mx.eval() calls are MLX's array materialization function, not Python eval().
"""

import mlx.core as mx
import mlx.nn as nn


def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None):
    """Sample next token from logits of shape (B, vocab_size). Returns (B, 1)."""
    if temperature == 0.0:
        return mx.argmax(logits, axis=-1, keepdims=True)

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.shape[-1])
        vals = mx.topk(logits, k=k, axis=-1)
        threshold = vals[:, -1:]
        logits = mx.where(logits >= threshold, logits, mx.array(float('-inf')))

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_indices = mx.argsort(logits, axis=-1)[:, ::-1]
        sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
        sorted_probs = mx.softmax(sorted_logits, axis=-1)
        cumulative = mx.cumsum(sorted_probs, axis=-1)
        # Mask tokens whose cumulative prob (excluding self) exceeds top_p
        mask = (cumulative - sorted_probs) > top_p
        sorted_logits = mx.where(mask, mx.array(float('-inf')), sorted_logits)
        # Unsort back to original order
        unsort_indices = mx.argsort(sorted_indices, axis=-1)
        logits = mx.take_along_axis(sorted_logits, unsort_indices, axis=-1)

    return mx.random.categorical(logits, axis=-1)[:, None]


class Engine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _materialize(self, *arrays):
        """Materialize MLX lazy arrays (wrapper around mx.eval)."""
        mx.eval(*arrays)

    def generate(self, tokens, num_samples=1, max_tokens=256, temperature=0.6, top_k=50, top_p=None):
        """Generate tokens with KV cache. Yields (token_column, token_masks) per step."""
        assert isinstance(tokens, list) and isinstance(tokens[0], int)

        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()

        # Prefill: run full prompt through model
        ids = mx.array([tokens])  # (1, T)
        logits, cache = self.model(ids)
        self._materialize(logits, *[c for pair in cache for c in pair])

        logits = logits[:, -1, :]  # (1, vocab_size)
        if num_samples > 1:
            logits = mx.broadcast_to(logits, (num_samples, logits.shape[-1]))

        completed = [False] * num_samples

        for _ in range(max_tokens):
            if all(completed):
                break

            next_ids = sample_next_token(logits, temperature, top_k, top_p)
            self._materialize(next_ids)
            sampled_tokens = next_ids[:, 0].tolist()

            token_column = []
            token_masks = []
            for i in range(num_samples):
                token = sampled_tokens[i]
                token_column.append(token)
                token_masks.append(1)
                if token == assistant_end or token == bos:
                    completed[i] = True

            yield token_column, token_masks

            # Decode step
            ids = next_ids
            logits, cache = self.model(ids, cache=cache)
            logits = logits[:, -1, :]
            self._materialize(logits, *[c for pair in cache for c in pair])

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """Non-streaming batch generation. Returns list of token sequences."""
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()

        results = [tokens.copy() for _ in range(num_samples)]
        completed = [False] * num_samples

        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
            if all(completed):
                break

        return results
