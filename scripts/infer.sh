#!/bin/bash

# Set paths
LLAMA_PATH="/path/to/llama"
LORA_PATH="/path/to/lora"
OUTPUT_FILE="/path/to/inference_results.jsonl"

# Inference parameters
MODE="qa"  # Change to "regeneration" for text regeneration
NUM_MEM=8
MAX_LENGTH=600
MAX_OUTPUT_LENGTH=600
MAX_CHUNKS=6
LORA_R=64
LORA_ALPHA=32
LORA_DROPOUT=0.2
PREDICTOR_ENABLED="--predictor_enabled"
LORA_ENABLED="--lora_enabled"
RATE=10

# Run inference
python ../infer.py \
    --mode $MODE \
    --llama_path $LLAMA_PATH \
    --lora_path $LORA_PATH \
    --output_file $OUTPUT_FILE \
    --num_mem $NUM_MEM \
    --max_length $MAX_LENGTH \
    --max_output_length $MAX_OUTPUT_LENGTH \
    --max_chunks $MAX_CHUNKS \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    $PREDICTOR_ENABLED \
    $LORA_ENABLED \
    --rate $RATE
