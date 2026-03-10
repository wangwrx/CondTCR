#!/bin/bash


source /home/user/anaconda3/bin/activate GRATCR

# Note: changed the data path
DATA_PATH="/home/user/path/to/CondTCR/Data/pairs/independent_clean.csv"
TRAIN_PATH="/home/user/path/to/CondTCR/Data/pairs/train.csv"
MODEL_PATH="/home/user/path/to/CondTCR/Model_results/CondTCR_aug.pth"

# Ensemble generation parameters (dual-ratio control)

# generation_ratios: Number of sequences to generate for each sampling method
# mixture_ratios: Proportion used to mix results from multiple sampling methods, should sum to 1 
# (Typically, generation_ratios are set greater than mixture_ratios to reduce duplicate samples after merging)

python /home/user/path/to/CondTCR/Codes/CondTCR_generate.py \
    --data_path ${DATA_PATH} \
    --train_data_path ${TRAIN_PATH} \
    --model_path ${MODEL_PATH} \
    --generation_mode ensemble \
    --ensemble_methods "beam,tkns" \
    --generation_ratios "1,1" \
    --mixture_ratios "0.5,0.5" \
    --num_return_sequences 100 \
    --num_beams 300 \
    --length_penalty 1.2 \
    --min_length 5 \
    --no_repeat_ngram_size 3 \
    --repetition_penalty 1.25 \
    --diversity_penalty 0.0 \
    --num_beam_groups 1 \
    --use_hf_beam \
    --initial_k 10 \
    --acs_q 8 \
    --top_p 0.92 \
    --top_k 60 \
    --batch_size 1 \
    --maxlen 32 \
    --random_seed 42




echo "Ensemble generation completed!"

