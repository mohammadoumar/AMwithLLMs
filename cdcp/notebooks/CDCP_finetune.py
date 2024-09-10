# *********** Finetune CDCP dataset for ACC, ARI and ARC tasks. ***************************

## ******* LIBRARIES: ********

# Run 7-25 only once to install LLaMA-Factory

# %cd ..
# %rm -rf LLaMA-Factory
# !git clone https://github.com/hiyouga/LLaMA-Factory.git
# %cd LLaMA-Factory
# %ls
# !pip install -e .[torch,bitsandbytes]

# !pip uninstall -y pydantic
# !pip install pydantic==1.10.9 # 

# !pip uninstall -y gradio
# !pip install gradio==3.48.0

# !pip uninstall -y bitsandbytes
# !pip install --upgrade bitsandbytes

# !pip install tqdm
# !pip install ipywidgets
# !pip install scikit-learn




import os
import ast
import sys
import json
import torch
import pickle
import inspect
import argparse
import subprocess

# sys.path.append('../')

import pandas as pd
from tqdm import tqdm
from pathlib import Path

from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc
from sklearn.metrics import classification_report


try:    
    assert torch.cuda.is_available() is True
    
except AssertionError:
    
    print("Please set up a GPU before using LLaMA Factory...")

current_dir = Path(os.path.dirname(os.path.realpath(__file__)))
cdcp_dir = current_dir.parent.absolute()
parent_dir = cdcp_dir.parent.absolute()
sys.path.append(os.path.abspath(cdcp_dir))

from utils.post_processing import *


## ********** Parameters: ************

parser = argparse.ArgumentParser()

parser.add_argument("model", help="The base model to fine-tune.", type=str)
parser.add_argument("task", help="The pipeline task to execute.", type=str, choices=["acc", "ari", "arc", "joint"])

args = parser.parse_args()

BASE_MODEL, TASK = args.model, args.task

ROOT_DIR = parent_dir #os.path.abspath(os.path.join(os.getcwd(), os.pardir))
DATASET_DIR = os.path.join(cdcp_dir, "datasets")
LLAMA_FACTORY_DIR = os.path.join(ROOT_DIR, "LLaMA-Factory")
OUTPUT_DIR = os.path.join(cdcp_dir, "finetuned_models_run3", f"""CDCP_{TASK}_{BASE_MODEL.split("/")[1]}""")


## ********** DATASET: ************


# *** TRAIN DATASET *** #

train_dataset_name = f"""CDCP_{TASK}_train.json"""
train_dataset_file = os.path.join(DATASET_DIR, train_dataset_name)

# *** TEST DATASET *** #

test_dataset_name = f"""CDCP_{TASK}_test.json"""
test_dataset_file = os.path.join(DATASET_DIR, test_dataset_name)


## ********** FINE-TUNE MODEL: ************


if not os.path.exists(os.path.join(cdcp_dir, "ft_arg_files")):
    os.mkdir(os.path.join(cdcp_dir, "ft_arg_files"))

# *** TRAIN FILE/DATASET INFO FILE ***

train_file = os.path.join(cdcp_dir, "ft_arg_files", f"""{train_dataset_name.split(".")[0].split("train")[0]}{BASE_MODEL.split("/")[1]}.json""")

dataset_info_line =  {
  "file_name": f"{train_dataset_file}",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output"
  }
}

with open(os.path.join(LLAMA_FACTORY_DIR, "data/dataset_info.json"), "r") as jsonFile:
    data = json.load(jsonFile)

data["cdcp"] = dataset_info_line

with open(os.path.join(LLAMA_FACTORY_DIR, "data/dataset_info.json"), "w") as jsonFile:
    json.dump(data, jsonFile)


# *** TRAINING ARGUMENTS ***

NB_EPOCHS = 5

args = dict(
  stage="sft",                           # do supervised fine-tuning
  do_train=True,
  model_name_or_path=BASE_MODEL,         # use bnb-4bit-quantized Llama-3-8B-Instruct model
  dataset="cdcp",           # use alpaca and identity datasets
  template="llama3",                     # use llama3 prompt template
  finetuning_type="lora",                # use LoRA adapters to save memory
  lora_target="all",                     # attach LoRA adapters to all linear layers
  output_dir=OUTPUT_DIR,                 # the path to save LoRA adapters
  overwrite_output_dir=True,             # overrides existing output contents
  per_device_train_batch_size=2,         # the batch size
  gradient_accumulation_steps=4,         # the gradient accumulation steps
  lr_scheduler_type="cosine",            # use cosine learning rate scheduler
  logging_steps=10,                      # log every 10 steps
  warmup_ratio=0.1,                      # use warmup scheduler
  save_steps=3000,                       # save checkpoint every 1000 steps
  learning_rate=5e-5,                    # the learning rate
  num_train_epochs=NB_EPOCHS,            # the epochs of training
  max_samples=2000,                       # use 500 examples in each dataset
  max_grad_norm=1.0,                     # clip gradient norm to 1.0
  quantization_bit=4,                    # use 4-bit QLoRA
  loraplus_lr_ratio=16.0,                # use LoRA+ algorithm with lambda=16.0
  fp16=True,                             # use float16 mixed precision training
  report_to="none"                       # discards wandb
)

json.dump(args, open(train_file, "w", encoding="utf-8"), indent=2)

p = subprocess.Popen(["llamafactory-cli", "train", train_file], cwd=LLAMA_FACTORY_DIR)

p.wait()


## ********** INFERENCE ON THE FINE-TUNED MODEL: ************

# *** LOADING SAVED MODEL ***

args = dict(
  model_name_or_path=BASE_MODEL, # use bnb-4bit-quantized Llama-3-8B-Instruct model
  adapter_name_or_path=OUTPUT_DIR,            # load the saved LoRA adapters
  template="llama3",                     # same to the one in training
  finetuning_type="lora",                  # same to the one in training
  quantization_bit=4,                    # load 4-bit quantized model
)

model = ChatModel(args)

# *** OBTAIN AND SAVE PREDICTIONS ***

with open(test_dataset_file, "r+") as fh:
    test_dataset = json.load(fh)

test_prompts = []
test_grounds = []

for sample in test_dataset:
    test_prompts.append("\nUser:" + sample["instruction"] + sample["input"])
    test_grounds.append(sample["output"])


test_predictions = []

for prompt in tqdm(test_prompts):

    messages = []
    messages.append({"role": "user", "content": prompt})

    response = ""
    
    for new_text in model.stream_chat(messages):
        #print(new_text, end="", flush=True)
        response += new_text
        #print()
    test_predictions.append({"role": "assistant", "content": response})

    torch_gc()


with open(os.path.join(OUTPUT_DIR, f"""CDCP_{TASK}_results_{NB_EPOCHS}.pickle"""), 'wb') as fh:
    results_d = {"ground_truths": test_grounds,
                 "predictions": test_predictions    
        
    }
    pickle.dump(results_d, fh)


## ********** POST-PROCESSING: ************

with open(os.path.join(OUTPUT_DIR, f"""CDCP_{TASK}_results_{NB_EPOCHS}.pickle"""), "rb") as fh:
        
        results = pickle.load(fh)

if TASK == 'acc':
    task_grounds, task_preds = post_process_acc(results)

elif TASK == 'ari':
    task_grounds, task_preds = post_process_ari(results)

elif TASK == 'arc':
    task_grounds, task_preds = post_process_arc(results)

elif TASK == 'joint':
    task_grounds, task_preds = post_process_joint(results)


print(classification_report(task_grounds, task_preds, digits=3))

with open(f"""{OUTPUT_DIR}/classification_report.pickle""", 'wb') as fh:
    
    pickle.dump(classification_report(task_grounds, task_preds, output_dict=True), fh)
