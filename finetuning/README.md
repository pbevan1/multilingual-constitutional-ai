# Constitutional AI 

## Full training examples

You will require about 350GB of VRAM to train the full model.
```shell
# Step 1 - SFT
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file finetuning/deepspeed_zero3.yaml finetuning/run_sft.py finetuning/sft/config.yaml

# Step 2 - DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file finetuning/deepspeed_zero3.yaml finetuning/run_dpo.py finetuning/dpo/config.yaml
# Note that we did not include the DPO recipe for grok, as that model's seems overtrained and too snarky.
```


## Advanced: generating you own dataset

To generate the constitutional AI dataset, see https://github.com/huggingface/llm-swarm/tree/main/examples/constitutional-ai for detailed instructions if you want to build or customize the dataset. 
