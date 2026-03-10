#!/bin/bash

cd /home/user/path/to/CondTCR/Codes

PYTHON_EXEC=/home/user/anaconda3/envs/CondTCR/bin/python
if [ ! -f "$PYTHON_EXEC" ]; then
  echo "Python interpreter not found at $PYTHON_EXEC"
  exit 1
fi

# other params
cal_chem="True"

model_path_save="/home/user/path/to/CondTCR/model/bert_pretrain.pth"

# execute benchmark script and good luck :)
"$PYTHON_EXEC" BERT.py \
  --data_path "/home/user/path/to/CondTCR/Data/pMHC/pMHC_36W_valid.csv" \
  --model_path ${model_path_save} \
  --epoch 60 \
  --maxlen 55 \
  --pMHC \

# Finish!
echo "Done!"