import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model
import torch.nn.functional as F

def load_lora_parameters(model, lora_params_path):
    """
    Initializes the LoRA parameters into the model.

    Args:
        model (AutoModelForCausalLM): Pretrained model with LoRA parameters.
        lora_params_path (str): Path to the saved LoRA parameters.
    """
    # Load the LoRA parameters from the provided file
    lora_params = torch.load(lora_params_path, map_location='cpu')
    
    # Initialize the LoRA parameters in the model
    with torch.no_grad():
        for name, param in model.named_parameters():
            print(name)
            if name in lora_params:  # If LoRA parameters exist for this name
                if 'lora' in name or 'memory_embeddings' in name:
                    param.copy_(lora_params[name])  # Copy the LoRA parameter
                    print(f"Loaded parameter for {name}")
                    param.requires_grad = True  # Enable gradient updates for this parameter

                elif "length_predictor" in name:
                    param.copy_(lora_params[name])
                    param.requires_grad = False  # Don't allow gradient updates for length predictor
                    print(f"Loaded parameter for {name} and set requires_grad to False")
                    
                else:
                    print(f"No saved parameter for {name}")
            elif "lora" in name:  # If no saved parameter, make the parameter trainable
                param.requires_grad = True
                print(f"No saved parameter for {name}")


class Probe(nn.Module):
    def __init__(self, hidden_dim=4096, seq_length=600, dropout_prob=0.2, num_heads=8, num_layers=2):
        """
        The Probe model is responsible for predicting the sequence length from hidden states.
        
        Args:
            hidden_dim (int): Dimension of the hidden states.
            seq_length (int): Length of the input sequence.
            dropout_prob (float): Dropout probability for regularization.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of attention layers.
        """
        super(Probe, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Dimensionality reduction layer (project to smaller dimension)
        self.dim_reduction = nn.Linear(hidden_dim, 1024)
        
        # Positional encoding for the input sequence
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_length, 1024))

        # Multi-head attention layers
        self.multihead_attention_layers = nn.ModuleList(
            [nn.MultiheadAttention(1024, num_heads) for _ in range(num_layers)]
        )
        
        # Fully connected layers for output processing
        self.fc1 = nn.Linear(1024 * seq_length, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 1)
        
        # Regularization layers
        self.dropout = nn.Dropout(p=dropout_prob)
        self.relu = nn.LeakyReLU(negative_slope=0.01) 
        self.layer_norm = nn.LayerNorm(1024)  

    def forward(self, hidden_states):
        """
        Forward pass through the probe model to predict the sequence length.

        Args:
            hidden_states (Tensor): Hidden states from the transformer model.
        
        Returns:
            length_pred (Tensor): Predicted sequence length.
        """
        # Apply dimensionality reduction and add positional encoding
        hidden_states = self.dim_reduction(hidden_states)
        hidden_states = hidden_states + self.positional_encoding

        # Pass through multi-head attention layers
        for layer in self.multihead_attention_layers:
            attn_output, _ = layer(hidden_states, hidden_states, hidden_states)
            hidden_states = self.layer_norm(attn_output + hidden_states)

        # Flatten the attention output for the fully connected layers
        attn_output_flat = attn_output.view(attn_output.size(0), -1)

        # Fully connected layers with residual connections
        x = self.fc1(attn_output_flat)
        x = self.relu(x)
        x = self.dropout(x)

        residual = x
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = F.layer_norm(x, normalized_shape=[x.size(-1)]) 
        x += residual

        # Further layers
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Final output for length prediction
        length_pred = self.fc5(x)

        return length_pred


class ATALoraQA(nn.Module):
    def __init__(self, llama_path, max_length, lora_config, num_mem, lora_path, max_output_length=600, 
                 max_chunks=6, predictor_enabled=True, lora_enabled=True, rate=10):
        """
        ATALoraQA integrates LoRA and memory mechanisms with a QA (Question-Answering) model.

        Args:
            llama_path (str): Path to the pretrained model.
            max_length (int): Maximum length of the input sequence.
            lora_config (LoraConfig): LoRA configuration to use.
            num_mem (int): Number of memory tokens to be used.
            lora_path (str): Path to the saved LoRA parameters.
            max_output_length (int): Maximum output length of the sequence.
            max_chunks (int): Number of chunks for memory handling.
            predictor_enabled (bool): Whether length prediction is enabled.
            lora_enabled (bool): Whether LoRA is enabled.
            rate (float): Scaling factor for dynamic memory size.
        """
        super(ATALoraQA, self).__init__()

        # Load the base Llama model and apply LoRA
        llama = AutoModelForCausalLM.from_pretrained(llama_path, torch_dtype=torch.bfloat16)
        self.llama = get_peft_model(llama, lora_config)

        # Freeze all parameters of the base Llama model
        for name, param in self.llama.named_parameters():
            param.requires_grad = False

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llama_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model configuration
        self.max_length = max_length
        self.num_mem = num_mem
        self.rate = rate
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # Memory embeddings (trainable if LoRA is enabled)
        self.memory_embeddings = nn.Parameter(torch.randn(1, num_mem, 4096, dtype=torch.bfloat16))
        self.memory_embeddings.requires_grad = lora_enabled

        # Length prediction configuration
        self.predictor_enabled = predictor_enabled
        self.lora_enabled = lora_enabled

        # Initialize the length predictor if enabled
        if self.predictor_enabled:
            self.length_predictor = Probe(hidden_dim=4096)
            self.length_criterion = nn.HuberLoss(delta=10)

        # Load LoRA parameters if specified
        if lora_path:
            load_lora_parameters(self, lora_path)

    def forward(self, input_ids, labels):
        """
        Forward pass for the ATALoraQA model. Processes input tokens and applies QA with memory.

        Args:
            input_ids (Tensor): Input token IDs.
            labels (Tensor): Target labels for training.
        
        Returns:
            dict: Contains 'loss' and 'logits'.
        """
        # Split input into text and QA tokens
        text_tokens = input_ids[:, :self.max_length]
        qa_tokens = input_ids[:, self.max_length:]
        target_tokens = labels

        # Get token embeddings for text and QA tokens
        text_tok_embeddings = self.llama.get_input_embeddings()(text_tokens)
        qa_tok_embeddings = self.llama.get_input_embeddings()(qa_tokens)

        # Pass through Llama encoder
        encoder_output = self.llama(inputs_embeds=text_tok_embeddings, output_hidden_states=True)
        hidden_states = encoder_output.hidden_states[-1]

        # Predict sequence length if enabled
        predicted_lengths = None
        if self.predictor_enabled:
            predicted_lengths = self.length_predictor(hidden_states).to(dtype=torch.bfloat16)
            nan_mask = torch.isnan(predicted_lengths)
            if nan_mask.any():
                print(f"Warning: NaN detected in predicted lengths, replacing with self.num_mem.")
                predicted_lengths[nan_mask] = 600.0  # Replace NaN values with default length

        # Dynamically adjust number of memory tokens based on predicted length
        dynamic_num_mem = (
            torch.round(predicted_lengths / self.rate).clamp(1, self.num_mem)
            if self.predictor_enabled else torch.tensor(self.num_mem)
        )
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

        # Trim past key values based on dynamic memory size
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

        # Prepare decoder inputs
        decoder_input_embeddings = qa_tok_embeddings

        # Run the decoder in evaluation mode
        with self.llama.disable_adapter():
            decoder_output = self.llama(
                inputs_embeds=decoder_input_embeddings, past_key_values=trimmed_past_key_values
            )

        # Calculate loss and return
        all_logits = decoder_output.logits
        loss = self.criterion(all_logits.view(-1, all_logits.size(-1)), target_tokens.view(-1))

        return {'loss': loss, 'logits': all_logits}
