# 📣 AMwithLLMs 📣

This repository contains the details of the project: **Argument Mining with Fine-Tuned Large Language Models**. *Fine-tuning* involves further training of a pre-trained model on a downstream dataset. This helps general-purpose LLL pre-training to be complemented with task specific supervised training.

<br>

# 📂 Repository Structure

This repository is organized as follows:

1) **abstRCT:** this directory contains the materiel for experiments on the Abstracts of Randomized Controlled Trials (AbstRCT) dataset.
1) **cdcp:** this directory contains the materiel for experiments on the Cornell eRulemaking Corpus (CDCP) dataset.
2) **mega:** this directory contains the materiel for implementation of a combined dataset consisting of all three datasets.
3) **pe:** this directory contains the materiel for experiments on the Persuasive Essays (PE) dataset.

```
.
├── abstRCT
├── cdcp
├── mega
└── pe
```

<br>

# ⛓️ Models

We experiment with the following models:

- **LLaMA-3-8B-Instruct** -- [**Meta AI**](meta-llama/Meta-Llama-3-8B-Instruct)
- **LLaMA-3-70B-Instruct** -- [**Meta AI**](meta-llama/Meta-Llama-3-70B-Instruct)
- **LLaMA-3.1-8B-Instruct** -- [**Meta AI**](meta-llama/Meta-Llama-3.1-8B-Instruct)

- **Gemma-2-9B-it** -- [**Google**](google/gemma-2-9b-it)
- **Qwen-2-7B-Instruct** -- [**Qwen**](Qwen/Qwen2-7B-Instruct)
- **Mistral-7B-Instruct** -- [**Mistral AI**](mistralai/Mistral-7B-Instruct-v0.3)
- **Phi-3-mini-instruct** -- [**Microsoft**](microsoft/Phi-3-mini-4k-instruct)

<br>

# 🎛️ Tasks

We experiment on the three tasks of an Argument Mining (AM) pipeline:

1) **Argument Component Classification (ACC):** ACC involves classifying an argument component as either *Major Claim*, *Claim* or *Premise*.
2) **Argument Relation Identification (ARI):** ARI involves classifying pairs of argument components as either *Related* or *Non-related*.
3) **Argument Relation Classification (ARC):** ARC involves classifying an argument relation as either *Support* or *Attack*.

<br>

# 📦 Requirements

We use the following versions of the packages:

```
torch==2.4.0
gradio==4.43.0
pydantic==2.9.0
LLaMA-Factory==0.9.0
transformers==4.44.2
bitsandbytes==0.43.1
```

<br>

# 💻 Platform and Compute

- For fine-tuning LLMs, we use [**LLaMA-Factory.**](https://github.com/hiyouga/LLaMA-Factory)
- For model checkpoints, we use [**Unsloth.**](https://unsloth.ai/)
- We also use [**Hugging Face.**](https://huggingface.co/)

All experiments have been performed on the High Performance Cluster at [**La Rochelle Université.**](https://www.univ-larochelle.fr/)
