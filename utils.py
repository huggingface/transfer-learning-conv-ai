# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import json
import logging
import multiprocessing as mp
import os
import tarfile
import tempfile
import torch

from pytorch_pretrained_bert import cached_path

PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"
HF_FINETUNED_MODEL = "https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/finetuned_chatbot_gpt.tar.gz"

logger = logging.getLogger(__file__)
logger.setLevel(level=logging.DEBUG)
mp.log_to_stderr(level=logging.DEBUG)
mp_logger = mp.get_logger()
mp_logger.setLevel(level=logging.DEBUG)


def download_pretrained_model():
    """ Download and extract finetuned model from S3 """
    resolved_archive_file = cached_path(HF_FINETUNED_MODEL)
    tempdir = tempfile.mkdtemp()

    logger.info("extracting archive file {} to temp dir {}".format(resolved_archive_file, tempdir))
    with tarfile.open(resolved_archive_file, 'r:gz') as archive:
        archive.extractall(tempdir)
    return tempdir


def worker_tokenize(args_list):
    """Target function for multiprocessing text encoding. All input args are included in a list as workaround
    for worker_tokenize() calling itself recursively with constant tokenizer as one argument.

    IMPORTANT: This function has to be implemented globally (outside of get_dataset()) to avoid
    multiprocessing error 'AttributeError: Can't pickle local object 'get_dataset.<locals>.worker_tokenize''.

    Args:
        args_list: [obj, tokenizer] as workaround for recursive self-calling of function within itself."""
    obj = args_list[0]
    tokenizer = args_list[1]
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        worker_tokenize._dict_key_calls += 1
        mp_logger.debug(
            'Encoding {}. obj.key() = {}, obj.items().__len__() = {}'.format(worker_tokenize._dict_key_calls,
                                                                             obj.keys(), obj.items().__len__()))
        return dict((n, worker_tokenize([o, tokenizer])) for n, o in obj.items())
    return list(worker_tokenize([o, tokenizer]) for o in obj)


worker_tokenize._dict_key_calls = 0


def get_dataset(tokenizer, dataset_path, dataset_cache=None):
    """ Get PERSONACHAT from S3 """
    dataset_path = dataset_path or PERSONACHAT_URL
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__  # Do avoid using GPT cache for GPT-2 and vice-versa
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Download dataset from %s", dataset_path)
        personachat_file = cached_path(dataset_path)
        with open(personachat_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())
        logger.info("Tokenize and encode the dataset")

        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                tokenize.dict_key_calls += 1
                logger.debug(
                    'Encoding {}. obj.keys() = {}, obj.items().__len__() = {}'.format(tokenize.dict_key_calls,
                                                                                      obj.keys(),
                                                                                      obj.items().__len__()))
                return dict((n, tokenize(o)) for n, o in obj.items())
            min_samples_for_multiprocessing = 100
            if obj.__len__() > min_samples_for_multiprocessing:
                logger.debug('  Encoding VERY LONG list of obj.__len__() = {}'.format(obj.__len__()))
                logger.debug('  Encoding list with with multiprocessing...')
                """functools.partial does not work becuase tokenizer has to be handed recusively together with obj to 
                worker_tokenize again. As a workaround of not knowing how to handle splash-operator for possible 
                dict-output and **kwargs input, the list_args is implemented."""
                with mp.Pool(processes=mp.cpu_count() - 1) as pool:
                    results = pool.map(func=worker_tokenize,
                                       iterable=[[o, tokenizer] for o in obj])
                return results
            else:
                logger.debug('  Encoding list of obj.__len__() = {}'.format(obj.__len__()))
                return list(tokenize(o) for o in obj)

        tokenize.dict_key_calls = 0

        dataset = tokenize(dataset)
        # dataset = tokenize(dataset)
        if dataset_cache:
            torch.save(dataset, dataset_cache)
    return dataset


def get_dataset_personalities(tokenizer, dataset_path, dataset_cache=None):
    """ Get personalities from PERSONACHAT """
    dataset_path = dataset_path or PERSONACHAT_URL
    dataset_cache = dataset_cache + '_' + type(tokenizer).__name__  # Do avoid using GPT cache for GPT-2 and vice-versa
    if os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        personachat = torch.load(dataset_cache)
    else:
        logger.info("Download PERSONACHAT dataset from %s", dataset_path)
        personachat_file = cached_path(dataset_path)
        with open(personachat_file, "r", encoding="utf-8") as f:
            personachat = json.loads(f.read())

        logger.info("Tokenize and encode the dataset")

        def tokenize(obj):
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)

        personachat = tokenize(personachat)
        # torch.save(personachat, dataset_cache)

    logger.info("Filter personalities")
    personalities = []
    for dataset in personachat.values():
        for dialog in dataset:
            personalities.append(dialog["personality"])

    logger.info("Gathered {} personalities".format(len(personalities)))
    return personalities


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
