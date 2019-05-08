# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import random
import logging
from pprint import pformat
from collections import defaultdict
from functools import partial
from tqdm import trange

import torch
import torch.nn.functional as F
from parlai.core.agents import Agent
from parlai.scripts.eval_model import setup_args as base_setup_args
from projects.convai2.eval_hits import eval_hits, setup_args as setup_args_hits
from projects.convai2.eval_f1 import eval_f1, setup_args as setup_args_f1
from projects.convai2.eval_ppl import eval_ppl, setup_args as setup_args_ppl
from projects.convai2.build_dict import build_dict
from pytorch_pretrained_bert import OpenAIGPTDoubleHeadsModel, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer

from train import build_input_from_segments, pad_dataset, SPECIAL_TOKENS
from utils import download_pretrained_model, AttrDict
from interact import sample_sequence

class TransformerAgent(Agent):
    @staticmethod
    def add_cmdline_args(argparser):
        agent_args = argparser.add_argument_group('Agent parameters')
        agent_args.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
        agent_args.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
        agent_args.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
        agent_args.add_argument("--eval_type", type=str, default="hits@1", help="hits@1, ppl or f1")
        agent_args.add_argument("--no_sample", action='store_true')
        agent_args.add_argument("--max_length", type=int, default=20)
        agent_args.add_argument("--min_length", type=int, default=1)
        agent_args.add_argument("--seed", type=int, default=0)
        agent_args.add_argument("--temperature", type=int, default=0.7)
        agent_args.add_argument("--top_k", type=int, default=20)
        return argparser

    def __init__(self, opt, shared=None):
        super(TransformerAgent, self).__init__(opt, shared)

        args = AttrDict(opt)  # to keep most commands identical to the interact.py script
        self.args = args

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__file__)
        self.logger.info(pformat(args))

        random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        if shared is None:
            self.logger.info("Get pretrained model and tokenizer")
            if args.model_checkpoint == "":
                args.model_checkpoint = download_pretrained_model()

            self.tokenizer = OpenAIGPTTokenizer.from_pretrained(args.model_checkpoint)
            if self.args.eval_type == "hits@1":
                self.model_checkpoint = OpenAIGPTDoubleHeadsModel.from_pretrained(args.model_checkpoint)
            else:
                self.model_checkpoint = OpenAIGPTLMHeadModel.from_pretrained(args.model_checkpoint)
            self.model_checkpoint.to(args.device)
            self.model_checkpoint.eval()

            self.logger.info("Build BPE prefix dictionary")
            convai_dict = build_dict()
            assert len(convai_dict) == 19304
            self.prefix2words = self.get_prefix2words(convai_dict)
        else:
            self.model_checkpoint = shared['model']
            self.tokenizer = shared['tokenizer']
            self.prefix2words = shared['prefix2words']

        self.special_tokens_ids = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

        self.persona = []
        self.history = []
        self.labels = []

        self.reset()

    def observe(self, observation):
        if self.episode_done:
            self.reset()

        if self.labels:
            # Add the previous response to the history
            self.history.append(self.labels)

        if 'labels' in observation or 'eval_labels' in observation:
            text = observation.get('labels', observation.get('eval_labels', [[]]))[0]
            self.labels = self.tokenizer.encode(text)

        if 'text' in observation:
            text = observation['text']
            for subtext in text.split('\n'):
                subtext = subtext.strip()
                if subtext.startswith('your persona:'):
                    subtext = subtext.replace('your persona:', '').strip()
                    self.persona.append(self.tokenizer.encode(subtext))
                else:
                    self.history.append(self.tokenizer.encode(subtext))

        self.history = self.history[-(2*self.args.max_history+1):]

        candidates = []
        if 'label_candidates' in observation:
            for candidate in observation['label_candidates']:
                candidates.append((self.tokenizer.encode(candidate), candidate))
        self.candidates = candidates

        self.episode_done = observation['episode_done']
        self.observation = observation
        return observation

    def act(self):
        reply = {}

        if self.args.eval_type == "hits@1" and len(self.candidates) > 0:
            instances = defaultdict(list)
            for candidate, _ in self.candidates:
                instance, _ = build_input_from_segments(self.persona, self.history, candidate, self.tokenizer)
                for input_name, input_array in instance.items():
                    instances[input_name].append(input_array)

            inputs = pad_dataset(instances, padding=self.special_tokens_ids[-1])

            tensor_inputs = {}
            for input_name in ["input_ids", "mc_token_ids", "token_type_ids"]:
                tensor = torch.tensor(inputs[input_name], device=self.args.device)
                tensor = tensor.view((-1, len(self.candidates)) + tensor.shape[1:])
                tensor_inputs[input_name] = tensor

            with torch.no_grad():
                _, mc_logits = self.model_checkpoint(**tensor_inputs)

            val, ind = torch.sort(mc_logits[0], descending=True)

            ypred = self.candidates[ind[0].item()][1] # match
            tc = []
            for j in range(len(self.candidates)):
                tc.append(self.candidates[ind[j].item()][1])
            reply = {'text': ypred, 'text_candidates': tc}
        else:
            # We are in interactive of f1 evaluation mode => just sample
            with torch.no_grad():
                out_ids, _ = sample_sequence(self.persona, self.history, self.tokenizer, self.model_checkpoint, self.args)
            out_text = self.tokenizer.decode(out_ids, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=(self.args.eval_type != 'f1'))
            reply = {'text': out_text}

        return reply

    def next_word_probability(self, partial_out):
        """Return probability distribution over next words given an input and
        partial true output. This is used to calculate the per-word perplexity.
        """
        partial_out_ids = self.tokenizer.encode(' '.join(partial_out))
        instance, _ = build_input_from_segments(self.persona, self.history, partial_out_ids,
                                             self.tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=self.args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=self.args.device).unsqueeze(0)

        with torch.no_grad():
            logits = self.model_checkpoint(input_ids, token_type_ids=token_type_ids)

        probs = F.softmax(logits[0, -1], dim=0)

        dist = {}
        for prefix_id, words in self.prefix2words.items():
            for word, ratio in words.items():
                dist[word] = probs[prefix_id].item() * ratio
        return dist

    def get_prefix2words(self, convai_dict, smoothing_freq=5):
        """ map BPE-prefix => dict(full_words beginning with BPE-prefix, associated words_counts) """
        prefix2words = defaultdict(dict)
        for i in trange(len(convai_dict)):
            word = convai_dict[i]
            freq = convai_dict.freq[word] + smoothing_freq
            bpe_tokens = self.tokenizer.bpe(word).split(' ')
            prefix_id = self.tokenizer.convert_tokens_to_ids(bpe_tokens[0])
            prefix2words[prefix_id].update(dict([(word, freq)]))

        for prefix_id, words in prefix2words.items():
            total_counts = sum(words.values())
            prefix2words[prefix_id] = dict((word, count/total_counts) for word, count in words.items())

        return prefix2words

    def share(self):
        shared = super(TransformerAgent, self).share()
        shared['tokenizer'] = self.tokenizer
        shared['model'] = self.model_checkpoint
        shared['prefix2words'] = self.prefix2words
        return shared

    def reset(self):
        self.persona = []
        self.history = []
        self.labels = []
        self.candidates = []
        self.episode_done = True
        self.observation = None


if __name__ == '__main__':
    parser = base_setup_args(None)
    parser.set_params(
        model='convai_evaluation:TransformerAgent')
    opt = parser.parse_args(print_args=False)

    if opt['eval_type'] == "hits@1":
        setup_args = setup_args_hits(None)
        eval_fct = partial(eval_hits, print_parser=setup_args)
    elif opt['eval_type'] == "ppl":
        setup_args = setup_args_ppl(None)
        eval_fct = eval_ppl
    elif opt['eval_type'] == "f1":
        setup_args = setup_args_f1(None)
        eval_fct = partial(eval_f1, print_parser=setup_args)
    else:
        raise ValueError

    setup_args.set_params(
        model='convai_evaluation:TransformerAgent')
    opt = setup_args.parse_args(print_args=False)

    eval_fct(opt)
