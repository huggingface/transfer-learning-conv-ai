from pathlib import Path
import shutil
import unittest

from transformers import OpenAIGPTTokenizer, GPT2Tokenizer
from train import ATTR_TO_SPECIAL_TOKEN, SPECIAL_TOKENS

class TestSpecialTokenTreatment(unittest.TestCase):

    def setUp(self):
        self.save_dir = Path('utest_save_dir')
        self.save_dir.mkdir(exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.save_dir)

    def test_special_tokens_checkpoint_behavior(self):
        toks = [OpenAIGPTTokenizer.from_pretrained('openai-gpt'), GPT2Tokenizer.from_pretrained('gpt2')]
        for tok in toks:
            self.assertEqual(len(tok.added_tokens_encoder), 0)
            tok.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
            self.assertEqual(len(tok.added_tokens_encoder), 5)
            # Make sure we never split
            self.assertEqual(len(tok.tokenize("<bos> <speaker1>")), 2)
            ids = tok.convert_tokens_to_ids(SPECIAL_TOKENS)
            self.assertTrue(all([x > 0 for x in ids]),
                            f'some tokens failed to tokenize {SPECIAL_TOKENS} -> {ids}')
            # Need to mantain indices through save. (this is also tested in pytorch-transformers)
            tok.save_pretrained(self.save_dir)
            tok_loaded = tok.from_pretrained(str(self.save_dir))
            ids2 = tok_loaded.convert_tokens_to_ids(SPECIAL_TOKENS)
            self.assertListEqual(ids, ids2)
