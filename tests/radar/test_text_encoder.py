"""The Hugging Face text encoder on a tiny local BERT snapshot (synthetic, built offline): a frozen encoder keeps its
evaluation behaviour in training mode, and a run resumes only with byte-identical text-model files."""

import os
import shutil
import tempfile
import unittest

import torch

from app.radar.model import HuggingFaceTextEncoder, build_model
from app.radar.records import write_records
from app.radar.train import run
from tests.radar.synthetic import ONTOLOGY, approved_clip_records
from tests.radar.test_training import Quiet, tiny_config

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "coupe", "apicale", "a4c", "mode", "b", "clip", ",", "."]


def bert_snapshot(path, dropout=0.3, seed=0):
    from transformers import BertConfig, BertModel, BertTokenizer
    from transformers.utils import logging

    logging.disable_progress_bar()
    os.makedirs(path)
    with open(os.path.join(path, "vocab.txt"), "w") as f:
        f.write("\n".join(VOCAB) + "\n")
    BertTokenizer(vocab_file=os.path.join(path, "vocab.txt")).save_pretrained(path)
    torch.manual_seed(seed)
    BertModel(BertConfig(vocab_size=len(VOCAB), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=32, max_position_embeddings=64, hidden_dropout_prob=dropout,
                         attention_probs_dropout_prob=dropout)).save_pretrained(path)
    return path


class TestHuggingFaceTextEncoder(Quiet):
    def setUp(self):
        super().setUp()
        self.snapshot = bert_snapshot(os.path.join(self.tmp, "bert"))

    def encode_twice(self, encoder):
        tokens = encoder.tokenize(["Coupe apicale A4C, mode B.", "Coupe A4C."])
        with torch.no_grad():
            return encoder(**tokens), encoder(**tokens)

    def test_a_frozen_encoder_has_no_dropout_in_training_mode(self):
        frozen = HuggingFaceTextEncoder(self.snapshot, freeze=True)
        frozen.train()
        self.assertFalse(frozen.model.training)
        first, second = self.encode_twice(frozen)
        self.assertTrue(torch.equal(first, second))
        # The model it belongs to does not switch it back either.
        cfg = tiny_config(self.tmp, "unused", **{"text.type": "huggingface", "text.name": self.snapshot,
                                                 "text.freeze": True})
        model, _ = build_model(cfg, 32, 4)
        model.train()
        self.assertFalse(model.text.model.training)
        self.assertTrue(model.video_proj.training and model.text_proj.training)

    def test_a_trained_encoder_keeps_dropout(self):
        trained = HuggingFaceTextEncoder(self.snapshot, freeze=False)
        trained.train()
        first, second = self.encode_twice(trained)
        self.assertFalse(torch.equal(first, second))  # so the frozen test above is not vacuous

    def test_files_hash_covers_weights_and_tokenizer_but_not_download_metadata(self):
        encoder = HuggingFaceTextEncoder(self.snapshot, freeze=True)
        digest = encoder.files_sha256()
        os.makedirs(os.path.join(self.snapshot, ".cache"))
        with open(os.path.join(self.snapshot, ".cache", "download.lock"), "w") as f:
            f.write("metadata")
        self.assertEqual(encoder.files_sha256(), digest)
        other = bert_snapshot(os.path.join(self.tmp, "other"), seed=1)
        for name in ("vocab.txt", "model.safetensors"):
            with self.subTest(name):
                copy = os.path.join(self.tmp, f"copy-{name}")
                shutil.copytree(self.snapshot, copy)
                if name == "vocab.txt":
                    with open(os.path.join(copy, name), "a") as f:
                        f.write("mitrale\n")
                else:
                    shutil.copy(os.path.join(other, name), copy)
                self.assertNotEqual(HuggingFaceTextEncoder(copy, freeze=True).files_sha256(), digest)

    def test_resume_refuses_changed_text_model_files(self):
        path = os.path.join(self.tmp, "records.jsonl")
        write_records(path, approved_clip_records(self.tmp, n=8, studies=4), ONTOLOGY)
        cfg = tiny_config(self.tmp, path, **{"text.type": "huggingface", "text.name": self.snapshot,
                                             "text.freeze": True, "text.max_length": 32})
        run(cfg, stop_after_steps=1)
        # Frozen text weights are not in latest.pt: a changed snapshot would silently change the model.
        bert_snapshot(os.path.join(self.tmp, "other"), seed=1)
        shutil.copy(os.path.join(self.tmp, "other", "model.safetensors"), self.snapshot)
        with self.assertRaisesRegex(ValueError, "text_files_sha256"):
            run(cfg)


if __name__ == "__main__":
    unittest.main()
