import os
import torch
from torch.utils.data import ConcatDataset
from transformers import TrainingArguments, Trainer
from finetune.dataset import TextDataset  # Create a separate dataset.py file
from peft import LoraConfig
from ATACompressorQA import ATALoraQA



def print_memory_usage(step):
    print(f"{step}: Allocated memory: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"{step}: Reserved memory: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

def create_datasets(ms_train_path, hqa_train_path, ms_test_path, hqa_test_path, llama_path, max_length, max_qa_length, limit_size_train=200000, limit_size_test=2000):
    ms_train_dataset = TextDataset(ms_train_path, llama_path, max_length, max_qa_length, limit_size=limit_size_train) 
    hqa_train_dataset = TextDataset(hqa_train_path, llama_path, max_length, max_qa_length, limit_size=limit_size_train)
    
    ms_test_dataset = TextDataset(ms_test_path, llama_path, max_length, max_qa_length, limit_size=limit_size_test)
    hqa_test_dataset = TextDataset(hqa_test_path, llama_path, max_length, max_qa_length, limit_size=limit_size_test)
    
    # Combine datasets for training and testing
    train_dataset = ConcatDataset([ms_train_dataset, hqa_train_dataset])
    test_dataset = ConcatDataset([ms_test_dataset, hqa_test_dataset])
    
    print("Datasets created.")
    print_memory_usage("AFTER DATA PROCESS")
    
    return train_dataset, test_dataset

def configure_lora(r=64, lora_alpha=32, lora_dropout=0.2):
    """Configure LoRA (Low-Rank Adaptation) parameters"""
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    return lora_config

def initialize_model(llama_path, max_length, lora_config, num_mem, lora_path):
    """Initialize model with LoRA configuration"""
    print("Loading llama + lora + llama ...")
    model = ATALoraQA(
        llama_path=llama_path,
        max_length=max_length,
        lora_config=lora_config,
        num_mem=num_mem,
        lora_path=lora_path,
    )
    
    print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model.config = model.llama.config
    print("llama + lora + llama loaded successfully.")
    print_memory_usage("AFTER MODEL CREATE")
    
    return model

def configure_training_args(output_dir, deepspeed_config, logging_dir, num_train_epochs, per_device_train_batch_size, 
                            per_device_eval_batch_size, save_strategy, save_steps, evaluation_strategy, eval_steps, 
                            warmup_steps):
    """Configure training parameters"""
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
    print_memory_usage("AFTER TRAINING ARGS")
    return training_args

def train_and_evaluate(model, training_args, train_dataset, test_dataset, resume_from_checkpoint):
    """Initialize trainer and start training"""
    # Disable automatic gradient anomaly detection for better performance
    torch.autograd.set_detect_anomaly(False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,        
    )

    # Resume training from checkpoint if provided, otherwise start from scratch
    if resume_from_checkpoint is None:
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Evaluate model after training
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
    max_qa_length = 300
    num_train_epochs = 4
    per_device_train_batch_size = 1
    per_device_eval_batch_size = 1
    save_strategy = "steps"
    save_steps = 1000
    evaluation_strategy = "steps"
    eval_steps = 1000
    warmup_steps = 300
    max_chunks = 6
    
    # Create datasets
    train_dataset, test_dataset = create_datasets(
        ms_train_path, hqa_train_path, ms_test_path, hqa_test_path,
        llama_path, max_length, max_qa_length
    )
    
    # Configure LoRA settings
    lora_config = configure_lora()
    
    # Initialize model
    model = initialize_model(llama_path, max_length, lora_config, num_mem, lora_path)
    
    # Configure training parameters
    training_args = configure_training_args(
        output_dir, deepspeed_config, logging_dir, num_train_epochs,
        per_device_train_batch_size, per_device_eval_batch_size, save_strategy,
        save_steps, evaluation_strategy, eval_steps, warmup_steps
    )
    
    # Start training and evaluation
    train_and_evaluate(model, training_args, train_dataset, test_dataset, resume_from_checkpoint)


