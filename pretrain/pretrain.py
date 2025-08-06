import os
import torch
from torch.utils.data import ConcatDataset
from transformers import TrainingArguments, Trainer
from pretrain.dataset import TextDataset 
from peft import LoraConfig
from ATACompressor import ATALora

# Clear CUDA cache and set environment variables for better memory management
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def create_datasets(ms_train_path, hqa_train_path, ms_test_path, hqa_test_path, llama_path, max_length, max_output_length):
    """Create training and evaluation datasets."""
    ms_train_dataset = TextDataset(ms_train_path, llama_path, max_length, max_output_length, limit_size=200000)
    hqa_train_dataset = TextDataset(hqa_train_path, llama_path, max_length, max_output_length, limit_size=200000)
    ms_test_dataset = TextDataset(ms_test_path, llama_path, max_length, max_output_length, limit_size=2000)
    hqa_test_dataset = TextDataset(hqa_test_path, llama_path, max_length, max_output_length, limit_size=2000)

    # Combine datasets for training and testing
    train_dataset = ConcatDataset([hqa_train_dataset, ms_train_dataset])
    test_dataset = ConcatDataset([hqa_test_dataset, ms_test_dataset])

    print("Datasets created.")
    return train_dataset, test_dataset

def configure_lora():
    """Configure LoRA (Low-Rank Adaptation) settings."""
    lora_config = LoraConfig(
        r=64,
        lora_alpha=32,
        lora_dropout=0.2,
        bias="none",
        task_type="CAUSAL_LM"
    )
    return lora_config

def initialize_model(llama_path, lora_config, max_length, max_output_length, num_mem, lora_path, max_chunks):
    """Initialize the model with LoRA configurations."""
    model = ATALora(
        llama_path=llama_path,
        max_length=max_length,
        lora_config=lora_config,
        num_mem=num_mem,
        lora_path=lora_path,
        max_output_length=max_output_length,
        max_chunks=max_chunks
    )

    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model.config = model.llama.config
    return model

def configure_training_args(output_dir, deepspeed_config, logging_dir, num_train_epochs, per_device_train_batch_size, 
                            per_device_eval_batch_size, save_strategy, save_steps, evaluation_strategy, eval_steps, warmup_steps):
    """Configure training arguments."""
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

def train_and_evaluate(model, training_args, train_dataset, test_dataset, resume_from_checkpoint):
    """Initialize trainer and start training."""
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

if __name__ == "__main__":
    # Configuration paths and parameters
    ms_train_path = "<to be filled>"
    ms_test_path = "<to be filled>"
    hqa_train_path = "<to be filled>"
    hqa_test_path = "<to be filled>"
    output_dir = "<to be filled>"
    llama_path = "<to be filled>"
    resume_from_checkpoint = "<to be filled>"
    deepspeed_config = "<to be filled>"
    logging_dir = "<to be filled>"
    lora_path = "<to be filled>"

    # Hyperparameters for training
    num_mem = 8
    max_length = 600
    max_output_length = 600
    num_train_epochs = 10
    per_device_train_batch_size = 1
    per_device_eval_batch_size = 1
    save_strategy = "steps"
    save_steps = 1000
    evaluation_strategy = "steps"
    eval_steps = 1000
    warmup_steps = 300
    max_chunks = 6

    # Create datasets
    train_dataset, test_dataset = create_datasets(ms_train_path, hqa_train_path, ms_test_path, hqa_test_path, 
                                                  llama_path, max_length, max_output_length)

    # Configure LoRA settings
    lora_config = configure_lora()

    # Initialize model
    compressor = initialize_model(llama_path, lora_config, max_length, max_output_length, num_mem, lora_path, max_chunks)

    # Configure training arguments
    training_args = configure_training_args(output_dir, deepspeed_config, logging_dir, num_train_epochs,
                                             per_device_train_batch_size, per_device_eval_batch_size, 
                                             save_strategy, save_steps, evaluation_strategy, eval_steps, warmup_steps)

    # Start training and evaluation
    train_and_evaluate(compressor, training_args, train_dataset, test_dataset, resume_from_checkpoint)
