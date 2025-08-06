import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model
import torch.nn.functional as F
import json


def load_lora_parameters(model, lora_params_path):
    """
    Initialize the LoRA parameters into the model.

    Args:
        model (AutoModelForCausalLM): The language model (LLM) with LoRA parameters.
        lora_params_path (str): Path to the saved LoRA parameters.
    """
    lora_params = torch.load(lora_params_path, map_location='cpu')
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in lora_params:
                if 'lora' in name or 'memory_embeddings' in name:
                    param.copy_(lora_params[name])
                    param.requires_grad = False
                elif "length_predictor" in name:
                    param.copy_(lora_params[name])
                    param.requires_grad = False
                else:
                    print(f"No saved parameter for {name}")
            elif "lora" in name:
                param.requires_grad = False


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
        self.dim_reduction = nn.Linear(hidden_dim, 1024)
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_length, 1024))
        self.multihead_attention_layers = nn.ModuleList(
            [nn.MultiheadAttention(1024, num_heads) for _ in range(num_layers)]
        )
        self.fc1 = nn.Linear(1024 * seq_length, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.layer_norm = nn.LayerNorm(1024)

    def forward(self, hidden_states):
        """
        Forward pass of the length predictor.

        Args:
            hidden_states (Tensor): The input hidden states from the model.

        Returns:
            length_pred (Tensor): The predicted length of the sequence.
        """
        hidden_states = self.dim_reduction(hidden_states)
        hidden_states = hidden_states + self.positional_encoding
        for layer in self.multihead_attention_layers:
            attn_output, _ = layer(hidden_states, hidden_states, hidden_states)
            hidden_states = self.layer_norm(attn_output + hidden_states)
        attn_output_flat = attn_output.view(attn_output.size(0), -1)
        x = self.fc1(attn_output_flat)
        x = self.relu(self.dropout(x))
        residual = x
        x = self.relu(self.dropout(F.layer_norm(self.fc2(x), normalized_shape=[x.size(-1)])) + residual)
        x = self.relu(self.dropout(self.fc3(x)))
        x = self.relu(self.dropout(self.fc4(x)))
        return self.fc5(x)


class ATALoraInfer(nn.Module):
    def __init__(self, llama_path, max_length, lora_config, num_mem, lora_path, max_output_length=600, max_chunks=6, 
                 predictor_enabled=True, lora_enabled=False, rate=10, output_file=None):
        """
        ATALoraInfer is a model for inference that integrates LoRA (Low-Rank Adaptation) with memory networks.

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
            output_file (str): Path to save the inference results.
        """
        super(ATALoraInfer, self).__init__()
        llama = AutoModelForCausalLM.from_pretrained(llama_path, torch_dtype=torch.bfloat16)
        self.llama = get_peft_model(llama, lora_config)
        for param in self.llama.parameters():
            param.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(llama_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        self.num_mem = num_mem
        self.rate = rate
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.memory_embeddings = nn.Parameter(torch.randn(1, num_mem, 4096, dtype=torch.bfloat16))
        self.memory_embeddings.requires_grad = lora_enabled
        self.predictor_enabled = predictor_enabled
        self.lora_enabled = lora_enabled
        self.output_file = output_file
        if self.predictor_enabled:
            self.length_predictor = Probe(hidden_dim=4096)
        if lora_path:
            load_lora_parameters(self, lora_path)

    def forward(self, input_ids, ground_truths, labels):
        """
        Forward pass for the ATALoraInfer model. Performs inference with memory handling.

        Args:
            input_ids (Tensor): Input token IDs.
            ground_truths (Tensor): Ground truth token IDs.
            labels (Tensor): Target labels for evaluation.

        Returns:
            dict: The loss dictionary, containing 'loss' as the total loss.
        """
        text_tok_embeddings = self.llama.get_input_embeddings()(input_ids)
        gold_tok_embeddings = self.llama.get_input_embeddings()(ground_truths)
        encoder_output = self.llama(inputs_embeds=text_tok_embeddings, output_hidden_states=True)
        hidden_states = encoder_output.hidden_states[-1]
        predicted_lengths = None
        if self.predictor_enabled:
            predicted_lengths = self.length_predictor(hidden_states).to(dtype=torch.bfloat16)
            predicted_lengths[torch.isnan(predicted_lengths)] = 600.0
        dynamic_num_mem = (
            torch.round(predicted_lengths / self.rate).clamp(1, self.num_mem)
            if self.predictor_enabled else torch.tensor(self.num_mem)
        )
        dynamic_num_mem = torch.where(dynamic_num_mem % 2 == 1, dynamic_num_mem + 1, dynamic_num_mem).clamp(2, self.num_mem)
        memory_tok_embeddings = torch.cat([
            self.memory_embeddings[:, :torch.max(torch.round(dynamic_num_mem[i]).int(), torch.tensor(2)).item(), :]
            for i in range(dynamic_num_mem.shape[0])
        ], dim=0)
        encoder_output = self.llama(
            inputs_embeds=memory_tok_embeddings,
            past_key_values=encoder_output.past_key_values,
            output_hidden_states=True
        )
        trimmed_past_key_values = tuple([
            (
                torch.cat([layer_key[i, :, -torch.round(dynamic_num_mem[i]).int():, :].unsqueeze(0) for i in range(dynamic_num_mem.shape[0])], dim=0),
                torch.cat([layer_value[i, :, -torch.round(dynamic_num_mem[i]).int():, :].unsqueeze(0) for i in range(dynamic_num_mem.shape[0])], dim=0)
            )
            for layer_key, layer_value in encoder_output.past_key_values
        ])
        prompt_tok_embeddings = self.llama.get_input_embeddings()(
            torch.tensor([self.tokenizer.bos_token_id], device=input_ids.device)
        ).repeat(gold_tok_embeddings.shape[0], 1, 1)
        decoder_input_embeddings = torch.cat((prompt_tok_embeddings, gold_tok_embeddings), dim=1)
        with self.llama.disable_adapter():
            decoder_output = self.llama(inputs_embeds=decoder_input_embeddings, past_key_values=trimmed_past_key_values)
        all_logits = decoder_output.logits
        loss = self.criterion(all_logits.view(-1, all_logits.size(-1)), labels.view(-1))

        # Save results to the output file
        if self.output_file:
            target_texts = [
                self.tokenizer.decode(labels[i][labels[i] != -100], skip_special_tokens=True)
                for i in range(labels.shape[0])
            ]
            target_lengths = [len(text.split()) for text in target_texts]
            results = [
                {
                    "target_length": target_lengths[i],
                    "predicted_length": float(predicted_lengths[i].item()),
                    "num_mem": int(dynamic_num_mem[i].item())
                }
                for i in range(len(target_lengths))
            ]
            with open(self.output_file, "a", encoding="utf-8") as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

        return {'loss': loss}
