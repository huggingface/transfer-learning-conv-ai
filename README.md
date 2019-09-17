# Edited for incorporating migration notes from pytorch-pretrained-bert to pytorch-transformers & added guidance for finetuning the pretrained dialog model ) to a custom dataset. 
----

- Author: Justin Cho 
- email: jcho at isi dot edu 
- [Original Repo](https://github.com/huggingface/transfer-learning-conv-ai)


1. If you are in a conda environment, you may need to run 'conda install -c conda-forge tensorflow tqdm' 
2. There are many issues with the original repo that require fixes to make it work as it says in the original README.
    - The pretrained model in the link is not completely compatible with `pytorch_transformers` You can use it to get pretrained weights for most layers, but you will need to further train it into a downstream task in order for the final decoder layer to function properly. Even if you make the migration fixes, running `python interact.py` will result in nonsense results, in the form: `<unk><unk>? ...`
    - An alternative to getting the desired results as shown in the [demo](https://convai.huggingface.co/) of the [Medium post](https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313) that goes along with the original repo is to train the model from scratch. This is not recommended as it will require a lot of computational resources to complete in a reasonable amount of time. 



# Outline of all the changes made for migration from pytorch-pretrained_bert

1. `import pytorch_pretrained_bert` replaced with `import pytorch_transformers`
2. `lm_loss, mc_loss = model(*batch)` in train.py to `lm_loss, mc_loss, lm_logits, mc_logits = model(*batch)` 
    - All model outputs are now in tuple format and the return values depend on each of the models. Check the documentations for more detail for each model. 
3. `SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]` to ```SPECIAL_TOKENS = {"bos_token": "<bos>", 
                  "eos_token": "<eos>",
                  "speaker1_token": "<speaker1>", 
                  "speaker2_token": "<speaker2>",
                  "pad_token": "<pad>"}```
4. `tb_logger.writer.log_dir` to `tb_logger.writer.logdir`: This is the correct attribute name as seen in the [tensorboardX docs](https://tensorboardx.readthedocs.io/en/latest/_modules/tensorboardX/writer.html#SummaryWriter)



# Steps for finetuning to custom dataset: 
At the current directory (~/transfer-learning-conv-ai/): 
1. Create directory to contain pretrained weights: `mkdir huggingface_s3; cd huggingface_s3`
2. Download the pretrained weights given in the original repo: `wget https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/finetuned_chatbot_gpt.tar.gz`
3. Extract weights: `tar -xvzf finetuned_chatbot_gpt.tar.gz`
4. Copy over files that contain information about special tokens: `cp special_tokens_map.json huggingface_s3/`, `cp added_tokens.json huggingface_s3/`  
5. Go to utils.py and change `CUSTOM_DATAPATH` to the custom data path. 
6. Change code content of `get_custom_dataset` function to reformat the custom dataset accordingly. 
7. Run `python train.py --model_checkpoint huggingface_s3/ --custom True `
8. You'll find your model checkpoints in `runs/` with their appropriate time stamps
9. Interact with the model by running `python interact.py --model_checkpoint runs/<path to checkpoint>/ --max_history 5 --max_length 100` 
    - I didn't want the personalities, so I commented them out. But you can always put them back in and comment out `personality=""` to incorporate the personalities. 
    - Setting the parameter values for `max_history` and `max_length` is optional. Without the personalities, the model can accommodate more previous dialog turns. 


changes that need to be made to pytorch_transformer.modeling_utils.py -> can be accommodated in train.py file 