#!/bin/bash

cd /home/user/path/to/CondTCR/Codes

PYTHON_EXEC=/home/user/anaconda3/envs/CondTCR/bin/python
if [ ! -f "$PYTHON_EXEC" ]; then
  echo "Python interpreter not found at $PYTHON_EXEC"
  exit 1
fi


model_path_save="/home/user/path/to/CondTCR/model_GPT/GPT_pretrain.pth"
train_samples=100000000

# execute benchmark script and good luck :)
"$PYTHON_EXEC" GPT.py \
  --data_path "/home/user/path/to/CondTCR/Data/TCR/beta.csv" \
  --model_path ${model_path_save} \
  --train_samples ${train_samples}

# Finish!
echo "Done!"