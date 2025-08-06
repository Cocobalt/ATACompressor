import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model
import torch.nn.functional as F

def load_lora_parameters(model, lora_params_path):
    """
    Initialize the LoRA parameters into the model.

    Args:
        model (AutoModelForCausalLM): The language model (LLM) with LoRA parameters.
        lora_params_path (str): Path to the saved LoRA parameters.
    """
    # Load the LoRA parameters from the provided path
    lora_params = torch.load(lora_params_path, map_location='cpu')
    
    # Initialize the LoRA parameters in the model and set the requires_grad flag
    with torch.no_grad():
        for name, param in model.named_parameters():
            print(name)
            if name in lora_params:  # Check if there is a LoRA parameter to load
                if 'lora' in name or 'memory_embeddings' in name:
                    param.copy_(lora_params[name])  # Copy the LoRA parameter
                    print(f"Loaded parameter for {name}")
                    param.requires_grad = True  # Allow gradient updates for this parameter

                elif "length_predictor" in name:  # Special handling for the length predictor parameters
                    param.copy_(lora_params[name])
                    param.requires_grad = True 
                    print(f"Loaded parameter for {name} and set requires_grad to True")
                    
                else:
                    print(f"No saved parameter for {name}")
            elif "lora" in name:  # If no parameter is saved, make the parameter trainable
                param.requires_grad = True
                print(f"No saved parameter for {name}")

class Probe(nn.Module):
    def __init__(self, hidden_dim=4096, seq_length=600, dropout_prob=0.2, num_heads=8, num_layers=2):
        """
        The Probe class predicts the length of the target sequence.
        
        Args:
            hidden_dim (int): The hidden dimension size for transformer layers.
            seq_length (int): The length of the input sequence.
            dropout_prob (float): Probability for applying dropout.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of multi-head attention layers.
        """
        super(Probe, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Dimensionality reduction layer
        self.dim_reduction = nn.Linear(hidden_dim, 1024)

        # Positional encoding for the input sequence
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_length, 1024))

        # Multi-head attention layers
        self.multihead_attention_layers = nn.ModuleList(
            [nn.MultiheadAttention(1024, num_heads) for _ in range(num_layers)]
        )
        
        # Fully connected layers for the final output
        self.fc1 = nn.Linear(1024 * seq_length, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 1)
        
        # Dropout and activation functions
        self.dropout = nn.Dropout(p=dropout_prob)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.layer_norm = nn.LayerNorm(1024)

    def forward(self, hidden_states):
        """
        Forward pass of the probe network.

        Args:
            hidden_states (Tensor): The input hidden states from the model.
        
        Returns:
            length_pred (Tensor): The predicted length of the sequence.
        """
        hidden_states = self.dim_reduction(hidden_states)
        hidden_states = hidden_states + self.positional_encoding  # Add positional encoding

        # Apply multi-head attention layers
        for layer in self.multihead_attention_layers:
            attn_output, _ = layer(hidden_states, hidden_states, hidden_states)
            hidden_states = self.layer_norm(attn_output + hidden_states)

        # Flatten the output and pass through fully connected layers
        attn_output_flat = attn_output.view(attn_output.size(0), -1)
        x = self.fc1(attn_output_flat)
        x = self.relu(x)
        x = self.dropout(x)

        # Residual connections for better training
        residual = x
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = F.layer_norm(x, normalized_shape=[x.size(-1)]) 
        x += residual

        # Further fully connected layers
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Final output for length prediction
        length_pred = self.fc5(x)
        return length_pred

class ATALora(nn.Module):
    def __init__(self, llama_path, max_length, lora_config, num_mem, lora_path, max_output_length=600, max_chunks=6, 
                 predictor_enabled=True, lora_enabled=True, rate=10):
        """
        ATALora is a model that integrates LoRA (Low-Rank Adaptation) with memory networks for sequence processing.

        Args:
            llama_path (str): Path to the pretrained model.
            max_length (int): Maximum length of input sequences.
            lora_config (LoraConfig): Configuration for LoRA.
            num_mem (int): Number of memory tokens to use.
            lora_path (str): Path to LoRA parameters.
            max_output_length (int): Maximum output length.
            max_chunks (int): Maximum number of chunks for memory handling.
            predictor_enabled (bool): Whether the length predictor is enabled.
            lora_enabled (bool): Whether LoRA is enabled.
            rate (float): Scaling factor for the dynamic memory number.
        """
        super(ATALora, self).__init__()
        
        # Load the LLM model with LoRA adapters
        llama = AutoModelForCausalLM.from_pretrained(llama_path, torch_dtype=torch.bfloat16)
        self.llama = get_peft_model(llama, lora_config)

        # Freeze all parameters of the base Llama model
        for name, param in self.llama.named_parameters():
            param.requires_grad = False
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llama_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Model hyperparameters
        self.max_length = max_length
        self.num_mem = num_mem
        self.rate = rate
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Memory embeddings for the model
        self.memory_embeddings = nn.Parameter(torch.randn(1, num_mem, 4096, dtype=torch.bfloat16))
        self.memory_embeddings.requires_grad = lora_enabled  # Set memory embeddings as trainable if LoRA is enabled

        # Flags for predictor and LoRA
        self.predictor_enabled = predictor_enabled
        self.lora_enabled = lora_enabled
        
        # Length prediction head (if enabled)
        if self.predictor_enabled:
            self.length_predictor = Probe(hidden_dim=4096)
            self.length_criterion = nn.HuberLoss(delta=10)

        # Load LoRA parameters if specified
        if lora_path:
            load_lora_parameters(self, lora_path)

    def forward(self, input_ids, ground_truths, labels):
        """
        Forward pass for the ATALora model. Performs sequence processing and memory handling.

        Args:
            input_ids (Tensor): Input token IDs.
            ground_truths (Tensor): Ground truth token IDs.
            labels (Tensor): Target labels for training.
        
        Returns:
            dict: The loss dictionary, containing 'loss' as the total loss.
        """
        text_tokens = input_ids
        target_tokens = labels
        gold_tokens = ground_truths

        # Embeddings for input and ground truth tokens
        text_tok_embeddings = self.llama.get_input_embeddings()(text_tokens)
        gold_tok_embeddings = self.llama.get_input_embeddings()(gold_tokens)

        # Pass through the Llama encoder
        encoder_output = self.llama(inputs_embeds=text_tok_embeddings, output_hidden_states=True)
        hidden_states = encoder_output.hidden_states[-1]

        # Predict sequence length if the predictor is enabled
        predicted_lengths = None
        if self.predictor_enabled:
            predicted_lengths = self.length_predictor(hidden_states).to(dtype=torch.bfloat16)
            nan_mask = torch.isnan(predicted_lengths)
            if nan_mask.any():
                print(f"Warning: NaN detected in predicted lengths, replacing with self.num_mem.")
                predicted_lengths[nan_mask] = 600.0
                
        # Calculate dynamic memory tokens based on predicted length
        dynamic_num_mem = (
            torch.round(predicted_lengths / self.rate).clamp(1, self.num_mem)
            if self.predictor_enabled else torch.tensor(self.num_mem)
        )
        
        # Ensure dynamic_num_mem is even and within bounds
        dynamic_num_mem = torch.where(dynamic_num_mem % 2 == 1, dynamic_num_mem + 1, dynamic_num_mem)
        dynamic_num_mem = dynamic_num_mem.clamp(2, self.num_mem)

        # Memory token embeddings
        memory_tok_embeddings = torch.cat([
            self.memory_embeddings[:, :torch.max(torch.round(dynamic_num_mem[i]).int(), torch.tensor(2)).item(), :].repeat(1, 1, 1)
            for i in range(dynamic_num_mem.shape[0])
        ], dim=0)
    
        # Re-run encoder with memory embeddings
        encoder_output = self.llama(
            inputs_embeds=memory_tok_embeddings,
            past_key_values=encoder_output.past_key_values, 
            output_hidden_states=True 
        )

        # Process past key values (hidden states)
        past_key_values = encoder_output.past_key_values

        # Trim the past key values based on dynamic memory
        trimmed_past_key_values = []
        for layer_key, layer_value in past_key_values:
            batch_trimmed_key = []
            batch_trimmed_value = []
            for i in range(dynamic_num_mem.shape[0]):
                num_mem = torch.round(dynamic_num_mem[i]).int()
                batch_trimmed_key.append(layer_key[i, :, -num_mem:, :].unsqueeze(0))
                batch_trimmed_value.append(layer_value[i, :, -num_mem:, :].unsqueeze(0))
            trimmed_past_key_values.append((
                torch.cat(batch_trimmed_key, dim=0), 
                torch.cat(batch_trimmed_value, dim=0)
            ))
        trimmed_past_key_values = tuple(trimmed_past_key_values)

        # Prepare prompt tokens and embeddings
        prompt_tokens = torch.tensor([self.tokenizer.bos_token_id], device=text_tokens.device)
        prompt_tok_embeddings = self.llama.get_input_embeddings()(prompt_tokens)
        prompt_tok_embeddings = prompt_tok_embeddings.repeat(gold_tok_embeddings.shape[0], 1, 1)

        # Concatenate prompt embeddings and gold token embeddings for decoding
        decoder_input_embeddings = torch.cat((prompt_tok_embeddings, gold_tok_embeddings), dim=1)

        # Run the decoder in evaluation mode
        with self.llama.disable_adapter():
            decoder_output = self.llama(
                inputs_embeds=decoder_input_embeddings, past_key_values=trimmed_past_key_values
            )

        # Calculate the logits and loss
        all_logits = decoder_output.logits
        loss = self.criterion(all_logits.view(-1, all_logits.size(-1)), target_tokens.view(-1))

        length_loss = 0
        if self.predictor_enabled:
            # Prepare target texts for length loss computation
            target_texts = [
                self.tokenizer.decode(labels[i][labels[i] != -100], skip_special_tokens=True)
                for i in range(labels.shape[0])
            ]
            target_lengths = torch.tensor(
                [len(text.split()) for text in target_texts], device=predicted_lengths.device,       
                dtype=torch.bfloat16, 
            ).unsqueeze(1)

            # Compute length loss
            length_loss = self.length_criterion(predicted_lengths, target_lengths).to(dtype=torch.bfloat16)
            
        # Total loss combining the prediction loss and length loss (if enabled)
        total_loss = loss + (0.00001 * length_loss if self.predictor_enabled else 0)

        return {
            'loss': total_loss,
        }
