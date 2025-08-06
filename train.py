import os
import torch
import argparse
from torch.utils.data import ConcatDataset
from transformers import TrainingArguments, Trainer
from dataset import TextDataset
from peft import LoraConfig
from pretrain.ATACompressor import ATALora
from finetune.ATACompressorQA import ATALoraQA

# Clear CUDA cache and set environment variables for better memory management
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def print_memory_usage(step):
    """Print current GPU memory usage."""
    print(f"{step}: Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"{step}: Reserved memory: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

def create_datasets(ms_train_path, hqa_train_path, ms_test_path, hqa_test_path, llama_path, 
                   max_length, max_output_length, mode, limit_size_train=200000, limit_size_test=2000):
    """
    Create training and evaluation datasets.
    
    Args:
        ms_train_path (str): Path to the MS Marco training dataset.
        hqa_train_path (str): Path to the HQA training dataset.
        ms_test_path (str): Path to the MS Marco test dataset.
        hqa_test_path (str): Path to the HQA test dataset.
        llama_path (str): Path to the pre-trained model.
        max_length (int): Maximum length of the input sequence.
        max_output_length (int): Maximum length of the output sequence.
        mode (str): 'pretrain' or 'finetune'.
        limit_size_train (int): Size limit for training dataset.
        limit_size_test (int): Size limit for test dataset.
        
    Returns:
        tuple: Training and testing datasets.
    """
    ms_train_dataset = TextDataset(ms_train_path, llama_path, max_length, max_output_length, 
                                  mode=mode, limit_size=limit_size_train)
    hqa_train_dataset = TextDataset(hqa_train_path, llama_path, max_length, max_output_length,
                                   mode=mode, limit_size=limit_size_train)
    
    ms_test_dataset = TextDataset(ms_test_path, llama_path, max_length, max_output_length,
                                 mode=mode, limit_size=limit_size_test)
    hqa_test_dataset = TextDataset(hqa_test_path, llama_path, max_length, max_output_length,
                                  mode=mode, limit_size=limit_size_test)

    # Combine datasets for training and testing
    train_dataset = ConcatDataset([hqa_train_dataset, ms_train_dataset])
    test_dataset = ConcatDataset([hqa_test_dataset, ms_test_dataset])

    print("Datasets created.")
    if mode == 'finetune':
        print_memory_usage("AFTER DATA PROCESS")
    
    return train_dataset, test_dataset

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

def initialize_model(llama_path, lora_config, max_length, max_output_length, num_mem, lora_path, 
                     mode='pretrain', max_chunks=6, predictor_enabled=True, lora_enabled=True, rate=10):
    """
    Initialize the model with LoRA configurations.
    
    Args:
        llama_path (str): Path to the pre-trained model.
        lora_config (LoraConfig): LoRA configuration.
        max_length (int): Maximum length of the input sequence.
        max_output_length (int): Maximum length of the output sequence.
        num_mem (int): Number of memory tokens.
        lora_path (str): Path to the saved LoRA parameters.
        mode (str): 'pretrain' or 'finetune'.
        max_chunks (int): Maximum number of chunks for memory.
        predictor_enabled (bool): Whether to enable length prediction.
        lora_enabled (bool): Whether to enable LoRA.
        rate (float): Rate for dynamic memory size.
        
    Returns:
        nn.Module: Initialized model.
    """
    if mode == 'pretrain':
        model = ATALora(
            llama_path=llama_path,
            max_length=max_length,
            lora_config=lora_config,
            num_mem=num_mem,
            lora_path=lora_path,
            max_output_length=max_output_length,
            max_chunks=max_chunks
        )
    else:  # finetune mode
        model = ATALoraQA(
            llama_path=llama_path,
            max_length=max_length,
            lora_config=lora_config,
            num_mem=num_mem,
            lora_path=lora_path,
            max_output_length=max_output_length,
            max_chunks=max_chunks,
            predictor_enabled=predictor_enabled,
            lora_enabled=lora_enabled,
            rate=rate
        )

    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model.config = model.llama.config
    
    return model

def configure_training_args(output_dir, deepspeed_config, logging_dir, num_train_epochs, per_device_train_batch_size, 
                           per_device_eval_batch_size, save_strategy, save_steps, evaluation_strategy, eval_steps, 
                           warmup_steps, mode='pretrain'):
    """
    Configure training arguments.
    
    Args:
        output_dir (str): Directory to save outputs.
        deepspeed_config (str): Path to DeepSpeed configuration.
        logging_dir (str): Directory for logs.
        num_train_epochs (int): Number of training epochs.
        per_device_train_batch_size (int): Batch size per device for training.
        per_device_eval_batch_size (int): Batch size per device for evaluation.
        save_strategy (str): Strategy for saving checkpoints.
        save_steps (int): Steps between checkpoint saves.
        evaluation_strategy (str): Strategy for evaluation.
        eval_steps (int): Steps between evaluations.
        warmup_steps (int): Number of warmup steps.
        mode (str): 'pretrain' or 'finetune'.
        
    Returns:
        TrainingArguments: Training arguments configuration.
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        save_strategy=save_strategy,
        save_steps=save_steps,
        gradient_accumulation_steps=4,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        eval_accumulation_steps=1,
        logging_dir=logging_dir,
        logging_steps=1,
        deepspeed=deepspeed_config,
        learning_rate=1e-5,
        save_total_limit=3,
        lr_scheduler_type='constant_with_warmup',
        warmup_steps=warmup_steps,
        log_level='debug',
        report_to=["none"],
        fp16=True,
        weight_decay=0.2,
        max_grad_norm=1.0
    )
    
    
    return training_args

def train_and_evaluate(model, training_args, train_dataset, test_dataset, resume_from_checkpoint, mode='pretrain'):
    """
    Initialize trainer and start training.
    
    Args:
        model (nn.Module): Model to train.
        training_args (TrainingArguments): Training arguments.
        train_dataset (Dataset): Training dataset.
        test_dataset (Dataset): Test dataset.
        resume_from_checkpoint (str): Path to checkpoint to resume from, or None.
        mode (str): 'pretrain' or 'finetune'.
    """
        # Disable automatic gradient anomaly detection for better performance
    torch.autograd.set_detect_anomaly(False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Resume training if checkpoint exists, otherwise train from scratch
    if resume_from_checkpoint is None:
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Evaluate the model after training
    evaluation_results = trainer.evaluate()
    print(f"Evaluation results: {evaluation_results}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a model for pretrain or finetune')
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['pretrain', 'finetune'], required=True, 
                        help='Mode: pretrain for compression or finetune for QA')
    
    # Paths
    parser.add_argument('--ms_train_path', type=str, required=True, 
                        help='Path to MS Marco training dataset')
    parser.add_argument('--hqa_train_path', type=str, required=True, 
                        help='Path to HQA training dataset')
    parser.add_argument('--ms_test_path', type=str, required=True, 
                        help='Path to MS Marco test dataset')
    parser.add_argument('--hqa_test_path', type=str, required=True, 
                        help='Path to HQA test dataset')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Directory to save outputs')
    parser.add_argument('--llama_path', type=str, required=True, 
                        help='Path to pretrained model')
    parser.add_argument('--lora_path', type=str, default=None, 
                        help='Path to saved LoRA parameters')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, 
                        help='Path to checkpoint to resume from')
    parser.add_argument('--deepspeed_config', type=str, required=True, 
                        help='Path to DeepSpeed configuration')
    parser.add_argument('--logging_dir', type=str, required=True, 
                        help='Directory for logs')
    
    # Hyperparameters
    parser.add_argument('--num_mem', type=int, default=8, 
                        help='Number of memory tokens')
    parser.add_argument('--max_length', type=int, default=600, 
                        help='Maximum length of input sequence')
    parser.add_argument('--max_output_length', type=int, default=600, 
                        help='Maximum length of output sequence')
    parser.add_argument('--num_train_epochs', type=int, default=10, 
                        help='Number of training epochs')
    parser.add_argument('--per_device_train_batch_size', type=int, default=1, 
                        help='Batch size per device for training')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1, 
                        help='Batch size per device for evaluation')
    parser.add_argument('--save_strategy', type=str, default="steps", 
                        help='Strategy for saving checkpoints')
    parser.add_argument('--save_steps', type=int, default=1000, 
                        help='Steps between checkpoint saves')
    parser.add_argument('--evaluation_strategy', type=str, default="steps", 
                        help='Strategy for evaluation')
    parser.add_argument('--eval_steps', type=int, default=1000, 
                        help='Steps between evaluations')
    parser.add_argument('--warmup_steps', type=int, default=300, 
                        help='Number of warmup steps')
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
    
    # Adjust default values based on mode
    if args.mode == 'finetune':
        if args.max_output_length == 600:  # If user didn't specify
            args.max_output_length = 300  # Default for finetune
        if args.num_train_epochs == 10:  # If user didn't specify
            args.num_train_epochs = 4  # Default for finetune
    
    # Create datasets
    train_dataset, test_dataset = create_datasets(
        args.ms_train_path, args.hqa_train_path, args.ms_test_path, args.hqa_test_path,
        args.llama_path, args.max_length, args.max_output_length, args.mode
    )
    
    # Configure LoRA settings
    lora_config = configure_lora(args.lora_r, args.lora_alpha, args.lora_dropout)
    
    # Initialize model
    model = initialize_model(
        args.llama_path, lora_config, args.max_length, args.max_output_length,
        args.num_mem, args.lora_path, args.mode, args.max_chunks,
        args.predictor_enabled, args.lora_enabled, args.rate
    )
    
    # Configure training parameters
    training_args = configure_training_args(
        args.output_dir, args.deepspeed_config, args.logging_dir, args.num_train_epochs,
        args.per_device_train_batch_size, args.per_device_eval_batch_size, args.save_strategy,
        args.save_steps, args.evaluation_strategy, args.eval_steps, args.warmup_steps, args.mode
    )
    
    # Start training and evaluation
    train_and_evaluate(model, training_args, train_dataset, test_dataset, args.resume_from_checkpoint, args.mode)
