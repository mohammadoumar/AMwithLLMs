#!/bin/bash

python3 PE_finetune.py unsloth/gemma-2-9b-it-bnb-4bit gemma acc 1 essay
python3 PE_finetune.py unsloth/gemma-2-9b-it-bnb-4bit gemma ari 0 paragraph
python3 PE_finetune.py unsloth/gemma-2-9b-it-bnb-4bit gemma arc 0 paragraph
python3 PE_finetune.py unsloth/Qwen2-7B-Instruct-bnb-4bit qwen acc 1 essay
python3 PE_finetune.py unsloth/Qwen2-7B-Instruct-bnb-4bit qwen ari 0 paragraph
python3 PE_finetune.py unsloth/Qwen2-7B-Instruct-bnb-4bit qwen arc 0 paragraph
python3 PE_finetune.py unsloth/Phi-3-mini-4k-instruct-bnb-4bit phi acc 1 essay
python3 PE_finetune.py unsloth/Phi-3-mini-4k-instruct-bnb-4bit phi ari 0 paragraph
python3 PE_finetune.py unsloth/Phi-3-mini-4k-instruct-bnb-4bit phi arc 0 paragraph
python3 PE_finetune.py unsloth/mistral-7b-instruct-v0.3-bnb-4bit mistral acc 1 essay
python3 PE_finetune.py unsloth/mistral-7b-instruct-v0.3-bnb-4bit mistral ari 0 paragraph
python3 PE_finetune.py unsloth/mistral-7b-instruct-v0.3-bnb-4bit mistral arc 0 paragraph