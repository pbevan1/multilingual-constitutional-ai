# Constitutional AI 

## Full training examples

You will require about 350GB of VRAM to train the full model. We used 8xA100 80GB.
```shell
# SFT (CAI)
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file finetuning/deepspeed_zero3.yaml finetuning/run_sft.py finetuning/sft/config.yaml

# DPO (CAI)
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file finetuning/deepspeed_zero3.yaml finetuning/run_dpo.py finetuning/dpo/config.yaml

# SFT (Baseline)
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file finetuning/deepspeed_zero3.yaml finetuning/run_sft.py finetuning/sft-baseline/config.yaml

```