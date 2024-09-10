
#!/bin/bash

# Define two lists
list1=("unsloth/llama-3-8b-Instruct-bnb-4bit" "unsloth/llama-3-8b-Instruct" "unsloth/llama-3-70b-Instruct-bnb-4bit")
# list1=("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit")
list2=("acc" "aric")

# Create an empty array to store argument pairs
arguments=()

# Generate argument pairs from the two lists

for item1 in "${list1[@]}"; do
    for item2 in "${list2[@]}"; do
        arguments+=("$item1 $item2")
    done
done

# Loop over the argument pairs and run the Python script with each
for args in "${arguments[@]}"; do
    echo "Running abstRCT_finetune.py with arguments: $args"

    python3 notebooks/abstRCT_finetune.py $args

    # Check the exit status of the Python script
    if [ $? -ne 0 ]; then
        echo -e "Error encountered with arguments: $args. Skipping to the next pair. \n \n  ************* \n"
        continue  # Skip to the next iteration
    fi

    echo -e  "Successfully ran abstRCT_finetune.py with arguments: $args \n \n  *************** \n"
done
