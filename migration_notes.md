### Incorporated migration notes from `pytorch-pretrained-bert` to `pytorch-transformers` for the [Original HuggingFace Repo](https://github.com/huggingface/transfer-learning-conv-ai)  & added guidance for finetuning the pretrained dialog model with a custom dataset. 
----

- Author: Justin Cho 
- [Original Repo](https://github.com/huggingface/transfer-learning-conv-ai)


1. If you are in a conda environment, you may need to run `conda install -c conda-forge tensorflow tqdm`  
2. There are many issues in the original repo due to the migration that require fixes to make it work as it is shown in the original repo's README. 
    - The pretrained model in the link is not completely compatible with `pytorch_transformers` You can use it to get pretrained weights for most layers, but you will need to further train it into a downstream task in order for the final decoder layer to function properly. Even if you make the migration fixes, running `python interact.py` will result in a chatbot with responses in the form: `<unk><unk>? ...`
    - An alternative to getting the desired results as shown in the [demo](https://convai.huggingface.co/) of the [Medium post](https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313) that goes along with the original repo is to train the model from scratch. This is not recommended as it will require a lot of computational resources to complete in a reasonable amount of time. 



# Outline of main changes made for migration from `pytorch_pretrained_bert`

1. `import pytorch_pretrained_bert` replaced with `import pytorch_transformers`
2. `lm_loss, mc_loss = model(*batch)` in train.py to `lm_loss, mc_loss, lm_logits, mc_logits = model(*batch)` 
    - All model outputs are now in tuple format and the return values depend on each of the models. Check the documentations for more detail for each model. 
3. `SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]` to ```SPECIAL_TOKENS = {"bos_token": "<bos>", 
                  "eos_token": "<eos>",
                  ~~"speaker1_token": "<speaker1>",~~
                  ~~"speaker2_token": "<speaker2>",~~
                  "additional_tokens": ["<speaker1>", "<speaker2>"],
                  "pad_token": "<pad>"}``` (Credits to [martinritchie](https://github.com/martinritchie) for pointing this out)
4. `tb_logger.writer.log_dir` to `tb_logger.writer.logdir`: This is the correct attribute name as seen in the [tensorboardX docs](https://tensorboardx.readthedocs.io/en/latest/_modules/tensorboardX/writer.html#SummaryWriter)
5. Other detailed changes can be found as comments in the source code. Most changes are simply applying the [migration notes](https://huggingface.co/pytorch-transformers/migration.html).


# Steps for finetuning with custom dataset: 
At the current directory (~/transfer-learning-conv-ai/): 
1. Create directory to contain pretrained weights: `mkdir huggingface_s3; cd huggingface_s3`
2. Download the pretrained weights given in the original repo: `wget https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/finetuned_chatbot_gpt.tar.gz`
3. Extract weights: `tar -xvzf finetuned_chatbot_gpt.tar.gz`
4. Copy over `special_tokens_map.json` that contain information about special tokens: `cp special_tokens_map.json huggingface_s3/`
5. Go to utils.py and change `CUSTOM_DATAPATH` to the custom data path. 
6. **Change code content of `get_custom_dataset` function to reformat the custom dataset accordingly.**
7. Run `python train.py --model_checkpoint huggingface_s3/ --custom True `
8. You'll find your model checkpoints in `runs/` with their appropriate time stamps
9. Interact with the model by running `python interact.py --model_checkpoint runs/<path to checkpoint>/` 


# Additional notes 

## Different key for weights in pretrained model 
When you run `train.py` or `interact.py` and get a warning similar to the following: 
`INFO:pytorch_transformers.modeling_utils:Weights of OpenAIGPTLMHeadModel not initialized from pretrained model: ['lm_head.weight']` 
and 
`INFO:pytorch_transformers.modeling_utils:Weights from pretrained model not used in OpenAIGPTLMHeadModel: ['multiple_choice_head.linear.weight', 'multiple_choice_head.linear.bias', 'lm_head.decoder.weight']`, and you would like to get these weights, you will have to manually match these weights despite their different names. I presume that these key changes in the `model.state_dict` happened some time during the migration and was not announced or I missed it. I added the following: 
```
       if key == 'lm_head.decoder.weight': 
                new_key = 'lm_head.weight'
            if key in ['multiple_choice_head.linear.weight', 'multiple_choice_head.linear.bias']: 
                new_key = key.replace('linear', 'summary')
```
to line 420 in `pytorch_transformers/modeling_utils.py` so that these name changes can be accommodated for. 

## Opt out of sampling personality 
Depending on your purpose of using this repo, you may or may not want to sample the personalities for your chatbot to refer to in generating responses when you run `python interact.py`. The defualt is set to using the personalities, but you can set the argument `--no_personality` to disable this. In this case, the model can better account for a longer dialog sequence, so you can set `--max_history` to a larger number. I found `--no_personality --max_history 5 --max_length 100` to work pretty well, although I haven't extensively experimented with other settings and cannot compare as it is difficult to automatically evaluate their results. 


