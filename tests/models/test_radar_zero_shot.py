# tests/models/test_radar_zero_shot.py

"""Zero-shot scoring: prompt-file refusals, the shipped prompt file, prompt-set pooling, the binary, categorical and
regression scorers on embeddings built to have a known answer, and encoding in order with the mode restored."""

import os
import tempfile
import unittest

import torch
import torch.nn.functional as F

from src.models.radar.text_encoder import HashTokenizer, TextEncoder, TinyTextBackbone
from src.models.radar.zero_shot import (
    binary_probability,
    categorical_probabilities,
    encode_prompts,
    load_prompt_tasks,
    pool,
    regression_estimate,
    regression_prompts,
)

PROMPTS = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "radar", "prompts_v0.yaml")


def write(text):
    handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    handle.write(text)
    handle.close()
    return handle.name


def unit(*xs):
    return F.normalize(torch.tensor(xs, dtype=torch.float32), dim=-1)


class TestPromptFile(unittest.TestCase):

    def test_the_shipped_file_loads_and_its_template_is_the_one_the_caption_builder_writes(self):
        import sys
        sys.path.append(os.path.join(os.path.dirname(PROMPTS), "..", "..", "data"))
        import radar_captions as rc
        task = load_prompt_tasks(PROMPTS)["lv_ejection_fraction"]
        self.assertEqual(task.mode, "regression")
        self.assertEqual(task.target, "lvef")                       # the column data/make_study_targets.py writes
        self.assertEqual(task.label_sources[0].replace("<#>", "{}"), rc.EF_SENTENCE)
        self.assertEqual(tuple(task.value_range), tuple(int(x) for x in rc.EF_RANGE))

    def test_refusals(self):
        cases = {
            "mode": "tasks:\n  a: {mode: ordinal, structure: lv, label_sources: [x]}\n",
            "negative_sources": "tasks:\n  a: {mode: binary, structure: lv, label_sources: [x]}\n",
            "one entry per label source": "tasks:\n  a: {mode: categorical, structure: lv, label_sources: [x, y], "
                                          "values: [1]}\n",
            "slot": "tasks:\n  a: {mode: regression, structure: lv, label_sources: [x], range: [0, 100]}\n",
            "range": "tasks:\n  a: {mode: regression, structure: lv, label_sources: ['<#>'], range: [5, 5]}\n",
            "structure": "tasks:\n  a: {mode: binary, label_sources: [x], negative_sources: [y]}\n",
            "non-empty": "tasks: {}\n",
            "target": "tasks:\n  a: {mode: binary, structure: lv, label_sources: [x], negative_sources: [y], "
                      "target: ''}\n",
        }
        for key, text in cases.items():
            path = write(text)
            self.addCleanup(os.remove, path)
            with self.subTest(key), self.assertRaisesRegex(ValueError, key):
                load_prompt_tasks(path)


    def test_the_target_is_the_task_name_unless_the_file_names_a_column(self):
        path = write("tasks:\n  a: {mode: binary, structure: lv, label_sources: [x], negative_sources: [y]}\n"
                     "  b: {mode: binary, structure: lv, label_sources: [x], negative_sources: [y], target: c}\n")
        self.addCleanup(os.remove, path)
        tasks = load_prompt_tasks(path)
        self.assertEqual((tasks["a"].target, tasks["b"].target), ("a", "c"))


class TestScorers(unittest.TestCase):

    def test_pool_is_the_normalised_mean(self):
        pooled = pool(torch.stack([unit(1, 0), unit(0, 1)]))
        self.assertTrue(torch.allclose(pooled, unit(1, 1)))
        with self.assertRaises(ValueError):
            pool(torch.zeros(0, 2))

    def test_binary_probability_follows_the_closer_prompt_set(self):
        positive = torch.stack([unit(1, 0.1), unit(1, -0.1)])
        negative = torch.stack([unit(0, 1), unit(0.1, 1)])
        z = torch.stack([unit(1, 0), unit(0, 1)])
        p = binary_probability(z, positive, negative, temperature=0.07)
        self.assertGreater(p[0].item(), 0.99)
        self.assertLess(p[1].item(), 0.01)
        swapped = binary_probability(z, negative, positive, temperature=0.07)
        self.assertTrue(torch.allclose(p + swapped, torch.ones(2), atol=1e-6))

    def test_pooling_modes_coincide_for_one_prompt_and_differ_for_a_spread_set(self):
        z = torch.stack([unit(1, 0.3), unit(0.2, 1)])
        one_p, one_n = unit(1, 0)[None], unit(0, 1)[None]
        a = binary_probability(z, one_p, one_n, 0.1, pooling="mean_similarity")
        b = binary_probability(z, one_p, one_n, 0.1, pooling="mean_embedding")
        self.assertTrue(torch.allclose(a, b, atol=1e-6))
        spread = torch.stack([unit(1, 0), unit(0, 1), unit(-1, 0.2)])         # a mean of unlike prompts is short
        tight = torch.stack([unit(0, 1), unit(0.05, 1)])
        by_similarity = binary_probability(z, spread, tight, 0.1, pooling="mean_similarity")
        by_embedding = binary_probability(z, spread, tight, 0.1, pooling="mean_embedding")
        self.assertFalse(torch.allclose(by_similarity, by_embedding, atol=1e-3))
        expected = torch.softmax(torch.stack([(z @ tight.T).mean(-1), (z @ spread.T).mean(-1)], -1) / 0.1, -1)[:, 1]
        self.assertTrue(torch.allclose(by_similarity, expected, atol=1e-6))
        with self.assertRaisesRegex(ValueError, "pooling"):
            binary_probability(z, spread, tight, 0.1, pooling="max")

    def test_categorical_probabilities_pick_the_nearest_grade_and_sum_to_one(self):
        grades = [torch.stack([unit(1, 0, 0)]), torch.stack([unit(0, 1, 0)]), torch.stack([unit(0, 0, 1)])]
        z = torch.stack([unit(0, 1, 0.1), unit(1, 0.1, 0)])
        p = categorical_probabilities(z, grades, temperature=0.05)
        self.assertEqual(p.argmax(-1).tolist(), [1, 0])
        self.assertTrue(torch.allclose(p.sum(-1), torch.ones(2), atol=1e-6))

    def test_regression_prompts_fill_the_slot_over_the_inclusive_range(self):
        pairs = regression_prompts(["EF <#> %.", "FE <#>"], 5, 7)
        self.assertEqual([v for v, _ in pairs], [5, 6, 7, 5, 6, 7])
        self.assertEqual(pairs[0][1], "EF 5 %.")
        self.assertEqual(pairs[-1][1], "FE 7")

    def test_regression_estimate_is_the_median_of_the_best_fraction(self):
        values = list(range(10))
        prompts = torch.eye(10)                                      # prompt v is axis v
        z = F.normalize(torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0.5, 1.0, 0.8]]), dim=-1)   # best axes 8, 9, 7
        # keep 0.3 of 10 prompts: the 3 most similar are values 8, 9, 7, whose median is 8
        self.assertEqual(regression_estimate(z, prompts, values, keep=0.3).item(), 8.0)
        # keep 0.1 keeps one: the single best value
        self.assertEqual(regression_estimate(z, prompts, values, keep=0.1).item(), 8.0)
        with self.assertRaises(ValueError):
            regression_estimate(z, prompts, values[:-1])
        with self.assertRaises(ValueError):
            regression_estimate(z, prompts, values, keep=0.0)

    def test_regression_with_several_templates_repeats_each_value(self):
        pairs = regression_prompts(["a <#>", "b <#>"], 0, 3)
        values = [v for v, _ in pairs]
        prompts = F.normalize(torch.randn(len(values), 6, generator=torch.Generator().manual_seed(0)), dim=-1)
        z = prompts[values.index(2)][None]                           # equal to the first prompt of value 2
        estimate = regression_estimate(z, prompts, values, keep=0.25)
        self.assertEqual(estimate.shape, (1,))
        self.assertEqual(estimate.item(), 2.0)


class TestEncodePrompts(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.encoder = TextEncoder(TinyTextBackbone(vocab_size=200, max_length=16), 32, proj_dim=8)
        self.tokenizer = HashTokenizer(200, 16)

    def test_embeddings_are_unit_norm_in_order_and_independent_of_the_batch_size(self):
        prompts = [f"fevg {v} %" for v in range(7)]
        a = encode_prompts(self.encoder, self.tokenizer, prompts, batch_size=3)
        b = encode_prompts(self.encoder, self.tokenizer, prompts, batch_size=7)
        self.assertEqual(a.shape, (7, 8))
        self.assertTrue(torch.allclose(a.norm(dim=-1), torch.ones(7), atol=1e-5))
        self.assertTrue(torch.allclose(a, b, atol=1e-5))
        self.assertTrue(torch.allclose(a[2], encode_prompts(self.encoder, self.tokenizer, [prompts[2]])[0], atol=1e-5))

    def test_the_encoders_mode_is_restored_and_no_gradient_is_kept(self):
        self.encoder.train()
        out = encode_prompts(self.encoder, self.tokenizer, ["fevg 40 %"])
        self.assertTrue(self.encoder.training)
        self.assertFalse(out.requires_grad)
        self.encoder.eval()
        encode_prompts(self.encoder, self.tokenizer, ["fevg 40 %"])
        self.assertFalse(self.encoder.training)


if __name__ == "__main__":
    unittest.main()
