#!/bin/bash

# Note: changed the code path

cd /homeuser/path/to/CondTCR/Codes

# Note: changed the Python interpreter path
PYTHON_EXEC=/home/user/anaconda3/envs/GRATCR/bin/python
if [ ! -f "$PYTHON_EXEC" ]; then
  echo "Python interpreter not found at $PYTHON_EXEC"
  exit 1
fi

# Note: changed the data path
data_path="/home/user/path/to/CondTCR/Data/pairs/train.csv"
bert_pt="/home/user/path/to/CondTCR/model/bert_pretrain.pth"
gpt_pt="/home/user/path/to/CondTCR/model/gpt_pretrain.pth"
model_path_save="/home/user/path/to/CondTCR/Model_results/CondTCR_aug.pth"

batch_size=32
learning_rate=3e-6
epoch=3

GPU="3"

Timestamp=$(date +%Y%m%d_%H%M%S)

# Data augmentation switches: to enable, append as needed:
# --enable_balanced_sampling --enable_conditional_noise --enable_conditional_dropout --enable_curriculum_learning --enable_batch_control

# execute and good luck :)
"$PYTHON_EXEC" CondTCR_train.py \
    --data_path ${data_path} \
    --bert_path ${bert_pt} \
    --gpt_path ${gpt_pt} \
    --model_path ${model_path_save} \
    --batch_size ${batch_size} \
    --learning_rate ${learning_rate} \
    --mode "train" \
    --epoch ${epoch} \
    --gpu ${GPU} \
    --freeze_gpt \
    --enable_balanced_sampling \
    --enable_conditional_dropout \
    --enable_curriculum_learning \
    --enable_batch_control


