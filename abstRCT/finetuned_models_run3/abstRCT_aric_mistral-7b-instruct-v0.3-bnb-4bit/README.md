---
license: other
library_name: peft
tags:
- llama-factory
- lora
- generated_from_trainer
base_model: unsloth/mistral-7b-instruct-v0.3-bnb-4bit
model-index:
- name: abstRCT_aric_mistral-7b-instruct-v0.3-bnb-4bit
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# abstRCT_aric_mistral-7b-instruct-v0.3-bnb-4bit

This model is a fine-tuned version of [unsloth/mistral-7b-instruct-v0.3-bnb-4bit](https://huggingface.co/unsloth/mistral-7b-instruct-v0.3-bnb-4bit) on the abstRCT dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 2
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 2
- gradient_accumulation_steps: 4
- total_train_batch_size: 16
- total_eval_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 5
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- PEFT 0.11.1
- Transformers 4.44.2
- Pytorch 2.3.0+cu121
- Datasets 2.19.2
- Tokenizers 0.19.1