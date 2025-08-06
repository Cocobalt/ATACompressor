#!/bin/bash

# Set paths
MS_TRAIN_PATH="/path/to/ms_train.json"
HQA_TRAIN_PATH="/path/to/hqa_train.json"
MS_TEST_PATH="/path/to/ms_test.json"
HQA_TEST_PATH="/path/to/hqa_test.json"
OUTPUT_DIR="/path/to/output"
LLAMA_PATH="/path/to/llama"
LORA_PATH="/path/to/lora"
DEEPSPEED_CONFIG="./config/deepspeed_configurations.json"
LOGGING_DIR="/path/to/logs"

# Training parameters
MODE="finetune"  # Change to "finetune" for fine-tuning
NUM_MEM=8
MAX_LENGTH=600
MAX_OUTPUT_LENGTH=600
NUM_TRAIN_EPOCHS=10
PER_DEVICE_TRAIN_BATCH_SIZE=1
PER_DEVICE_EVAL_BATCH_SIZE=1
SAVE_STRATEGY="steps"
SAVE_STEPS=1000
EVALUATION_STRATEGY="steps"
EVAL_STEPS=1000
WARMUP_STEPS=300
MAX_CHUNKS=6
LORA_R=64
LORA_ALPHA=32
LORA_DROPOUT=0.2
PREDICTOR_ENABLED="--predictor_enabled"
LORA_ENABLED="--lora_enabled"
RATE=10

# Run training
python ../train.py \
    --mode $MODE \
    --ms_train_path $MS_TRAIN_PATH \
    --hqa_train_path $HQA_TRAIN_PATH \
    --ms_test_path $MS_TEST_PATH \
    --hqa_test_path $HQA_TEST_PATH \
    --output_dir $OUTPUT_DIR \
    --llama_path $LLAMA_PATH \
    --lora_path $LORA_PATH \
    --deepspeed_config $DEEPSPEED_CONFIG \
    --logging_dir $LOGGING_DIR \
    --num_mem $NUM_MEM \
    --max_length $MAX_LENGTH \
    --max_output_length $MAX_OUTPUT_LENGTH \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --save_strategy $SAVE_STRATEGY \
    --save_steps $SAVE_STEPS \
    --evaluation_strategy $EVALUATION_STRATEGY \
    --eval_steps $EVAL_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --max_chunks $MAX_CHUNKS \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    $PREDICTOR_ENABLED \
    $LORA_ENABLED \
    --rate $RATE
