import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from kv_cache import KVCompressor


class FakeTokenizer:
    def decode(self, tokens, skip_special_tokens=True):
        return tokens


def tiny_model():
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        attn_implementation="eager",
        use_sliding_window=False,
    )
    return Qwen2ForCausalLM(config).eval()


class Compression(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.compressor = KVCompressor(self.model, FakeTokenizer(), window=4, kernel=3)
        self.tokens = torch.randint(0, 100, (40,), generator=torch.Generator().manual_seed(1)).tolist()
        self.prefilled = self.compressor.prefill(self.tokens)

    def test_full_budget_matches_transformers_generate(self):
        expected = self.model.generate(
            torch.tensor([self.tokens]), max_new_tokens=8, do_sample=False, pad_token_id=0, eos_token_id=None,
            temperature=None, top_p=None, top_k=None,
        )[0, len(self.tokens) :].tolist()
        output = self.compressor.generate(self.prefilled, [40] * 4, max_new_tokens=8)
        self.assertEqual(output, expected)

    def test_each_layer_keeps_its_own_budget(self):
        cache = self.compressor._compress(self.prefilled, [4, 10, 20, 7])
        self.assertEqual([layer.keys.shape[2] for layer in cache.layers], [4, 10, 20, 7])

    def test_window_is_always_retained(self):
        for budget, expected in [(1, 4), (4, 4), (6, 6)]:
            cache = self.compressor._compress(self.prefilled, [budget] * 4)
            self.assertEqual(cache.layers[0].keys.shape[2], expected)
            self.assertTrue(torch.equal(cache.layers[0].keys[:, :, -4:], self.prefilled.keys[0][:, :, -4:]))

    def test_different_budgets_per_layer_can_decode(self):
        output = self.compressor.generate(self.prefilled, [4, 30, 8, 12], max_new_tokens=5)
        self.assertEqual(len(output), 5)

    def test_order_covers_context_without_window(self):
        for order in self.prefilled.order:
            self.assertEqual(sorted(order.tolist()), list(range(36)))


if __name__ == "__main__":
    unittest.main()
