import os
import torch
import argparse
from transformers import TrainingArguments
from peft import LoraConfig
from inference.QA import ATALoraQAInfer
from inference.Regeneration import ATALoraInfer

# Clear CUDA cache and set environment variables for better memory management
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def configure_lora(r=64, lora_alpha=32, lora_dropout=0.2):
    """
    Configure LoRA (Low-Rank Adaptation) settings.
    
    Args:
        r (int): Rank of the LoRA adaptation.
        lora_alpha (int): Scaling factor for the LoRA adaptation.
        lora_dropout (float): Dropout probability for LoRA layers.
        
    Returns:
        LoraConfig: LoRA configuration object.
    """
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    return lora_config

def initialize_infer_model(llama_path, lora_config, max_length, max_output_length, num_mem, lora_path, 
                           mode='qa', max_chunks=6, predictor_enabled=True, lora_enabled=True, rate=10, output_file=None):
    """
    Initialize the inference model with LoRA configurations.
    
    Args:
        llama_path (str): Path to the pre-trained model.
        lora_config (LoraConfig): LoRA configuration.
        max_length (int): Maximum length of the input sequence.
        max_output_length (int): Maximum length of the output sequence.
        num_mem (int): Number of memory tokens.
        lora_path (str): Path to the saved LoRA parameters.
        mode (str): 'qa' for QA inference or 'regeneration' for text regeneration.
        max_chunks (int): Maximum number of chunks for memory.
        predictor_enabled (bool): Whether to enable length prediction.
        lora_enabled (bool): Whether to enable LoRA.
        rate (float): Rate for dynamic memory size.
        output_file (str): Path to save the inference results.
        
    Returns:
        nn.Module: Initialized inference model.
    """
    if mode == 'qa':
        model = ATALoraQAInfer(
            llama_path=llama_path,
            max_length=max_length,
            lora_config=lora_config,
            num_mem=num_mem,
            lora_path=lora_path,
            max_output_length=max_output_length,
            max_chunks=max_chunks,
            predictor_enabled=predictor_enabled,
            lora_enabled=lora_enabled,
            rate=rate,
            output_file=output_file
        )
    else:  # regeneration mode
        model = ATALoraInfer(
            llama_path=llama_path,
            max_length=max_length,
            lora_config=lora_config,
            num_mem=num_mem,
            lora_path=lora_path,
            max_output_length=max_output_length,
            max_chunks=max_chunks,
            predictor_enabled=predictor_enabled,
            lora_enabled=lora_enabled,
            rate=rate,
            output_file=output_file
        )

    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    return model

def parse_arguments():
    """Parse command line arguments for inference."""
    parser = argparse.ArgumentParser(description='Run inference for QA or text regeneration tasks')
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['qa', 'regeneration'], required=True, 
                        help='Mode: qa for question answering or regeneration for text generation')
    
    # Paths
    parser.add_argument('--llama_path', type=str, required=True, 
                        help='Path to pretrained model')
    parser.add_argument('--lora_path', type=str, default=None, 
                        help='Path to saved LoRA parameters')
    parser.add_argument('--output_file', type=str, required=True, 
                        help='Path to save inference results')
    
    # Hyperparameters
    parser.add_argument('--num_mem', type=int, default=8, 
                        help='Number of memory tokens')
    parser.add_argument('--max_length', type=int, default=600, 
                        help='Maximum length of input sequence')
    parser.add_argument('--max_output_length', type=int, default=600, 
                        help='Maximum length of output sequence')
    parser.add_argument('--max_chunks', type=int, default=6, 
                        help='Maximum number of chunks for memory')
    parser.add_argument('--lora_r', type=int, default=64, 
                        help='Rank of LoRA adaptation')
    parser.add_argument('--lora_alpha', type=int, default=32, 
                        help='Scaling factor for LoRA adaptation')
    parser.add_argument('--lora_dropout', type=float, default=0.2, 
                        help='Dropout probability for LoRA layers')
    parser.add_argument('--predictor_enabled', action='store_true', 
                        help='Enable length prediction')
    parser.add_argument('--lora_enabled', action='store_true', 
                        help='Enable LoRA')
    parser.add_argument('--rate', type=float, default=10, 
                        help='Rate for dynamic memory size')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure LoRA settings
    lora_config = configure_lora(args.lora_r, args.lora_alpha, args.lora_dropout)
    
    # Initialize inference model
    model = initialize_infer_model(
        llama_path=args.llama_path,
        lora_config=lora_config,
        max_length=args.max_length,
        max_output_length=args.max_output_length,
        num_mem=args.num_mem,
        lora_path=args.lora_path,
        mode=args.mode,
        max_chunks=args.max_chunks,
        predictor_enabled=args.predictor_enabled,
        lora_enabled=args.lora_enabled,
        rate=args.rate,
        output_file=args.output_file
    )
    
    print("Inference model initialized. Ready to process input data.")
