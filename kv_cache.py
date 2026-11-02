"""Kompresi KV cache ala SnapKV dengan budget berbeda di tiap layer.

Satu sampel hanya perlu satu kali prefill. Hasilnya (KV penuh dan urutan
kepentingan token per layer) disimpan, lalu setiap kandidat budget cukup memilih
token dari sana dan melakukan decoding. Inilah yang membuat GA cukup murah.

Forward ditulis sendiri atas `model.model.layers` karena mask bawaan transformers
mengasumsikan semua layer punya panjang cache yang sama. Decoding memasukkan satu
token per langkah, jadi tidak butuh mask sama sekali.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import DynamicCache


@dataclass
class Prefilled:
    keys: list[torch.Tensor]  # per layer, di CPU
    values: list[torch.Tensor]
    order: list[torch.Tensor]  # indeks token konteks per layer, terpenting lebih dulu
    first_logits: torch.Tensor
    length: int


class KVCompressor:
    """Model harus dimuat dengan attn_implementation="eager" agar bobot atensi tersedia."""

    def __init__(self, model, tokenizer, window: int = 32, kernel: int = 7):
        self.model = model
        self.tokenizer = tokenizer
        self.window = window
        self.kernel = kernel
        self.device = next(model.parameters()).device

    def _forward(self, input_ids, cache, position_ids, mask=None):
        backbone = self.model.model
        hidden = backbone.embed_tokens(input_ids)
        position_embeddings = backbone.rotary_emb(hidden, position_ids)
        for layer in backbone.layers:
            hidden = layer(
                hidden,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_values=cache,
                position_embeddings=position_embeddings,
            )
        return self.model.lm_head(backbone.norm(hidden[:, -1]))

    def _importance(self, attention: torch.Tensor) -> torch.Tensor:
        """Atensi dari token window ke konteks, dirata-rata per head dan query, lalu max pool."""
        window_attention = attention[:, :, -self.window :, : -self.window]
        scores = window_attention.float().mean(dim=(1, 2))
        pooled = F.max_pool1d(scores, self.kernel, stride=1, padding=self.kernel // 2)
        return pooled[0].argsort(descending=True).cpu()

    @torch.inference_mode()
    def prefill(self, token_ids: list[int]) -> Prefilled:
        input_ids = torch.tensor([token_ids], device=self.device)
        length = input_ids.shape[1]
        if length <= self.window:
            raise ValueError(f"prompt ({length} token) harus lebih panjang dari window ({self.window})")

        order = []

        def capture(_module, _args, output):
            order.append(self._importance(output[1]))

        hooks = [layer.self_attn.register_forward_hook(capture) for layer in self.model.model.layers]
        dtype = next(self.model.parameters()).dtype
        mask = torch.full((length, length), torch.finfo(dtype).min, dtype=dtype, device=self.device)
        mask = torch.triu(mask, diagonal=1)[None, None]
        position_ids = torch.arange(length, device=self.device)[None]
        cache = DynamicCache()
        try:
            first_logits = self._forward(input_ids, cache, position_ids, mask)
        finally:
            for hook in hooks:
                hook.remove()

        keys = [layer.keys.cpu() for layer in cache.layers]
        values = [layer.values.cpu() for layer in cache.layers]
        return Prefilled(keys, values, order, first_logits.cpu(), length)

    def _compress(self, prefilled: Prefilled, budgets: list[int]) -> DynamicCache:
        """Budget mencakup window: k_i = token terpilih + token window."""
        window_ids = torch.arange(prefilled.length - self.window, prefilled.length)
        retained = []
        for key, value, order, budget in zip(prefilled.keys, prefilled.values, prefilled.order, budgets):
            selected = order[: max(budget - self.window, 0)].sort().values
            index = torch.cat([selected, window_ids])
            retained.append((key[:, :, index].to(self.device), value[:, :, index].to(self.device)))
        return DynamicCache(retained)

    @torch.inference_mode()
    def generate(self, prefilled: Prefilled, budgets: list[int], max_new_tokens: int = 40) -> str:
        if len(budgets) != len(prefilled.keys):
            raise ValueError(f"butuh {len(prefilled.keys)} budget, diterima {len(budgets)}")

        cache = self._compress(prefilled, budgets)
        eos = self.model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        logits = prefilled.first_logits.to(self.device)
        tokens = []
        for step in range(max_new_tokens):
            token = logits.argmax(dim=-1, keepdim=True)
            if token.item() in eos:
                break
            tokens.append(token.item())
            position_ids = torch.tensor([[prefilled.length + step]], device=self.device)
            logits = self._forward(token, cache, position_ids)
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
