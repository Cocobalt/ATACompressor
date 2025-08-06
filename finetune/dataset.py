import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

def read_jsonl_file(file_path):
    """
    Load JSONL file and return a list of JSON objects (dictionaries).

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        List[dict]: List of dictionaries parsed from the JSONL file.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))  # Parse each line as a JSON object
    return data

class TextDataset(Dataset):
    def __init__(self, text_file, llama_path, max_length, max_qa_length, limit_size=None):
        """
        Initialize the dataset for text processing.

        Args:
            text_file (str): Path to the input text file (JSONL format).
            llama_path (str): Path to the pre-trained model.
            max_length (int): Maximum length of the input sequence.
            max_qa_length (int): Maximum length of the QA sequence.
            limit_size (int, optional): Limit the number of samples in the dataset (default is None).
        """
        self.text = read_jsonl_file(text_file)
        if limit_size:
            self.text = self.text[:limit_size]
            
        self.tokenizer = AutoTokenizer.from_pretrained(llama_path, use_auth_token="<to be filled>")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_context_length = max_length
        self.num_mem = 8  # Can be passed as a parameter
        self.eos_tok_id = self.tokenizer.eos_token_id
        
        self.max_qa_len = max_qa_length
        # Preprocess all data
        self.preprocessed_data = self._preprocess_data()
        print(f"Total dataset size: {len(self.preprocessed_data)}")
        
    def _preprocess_data(self):
        """
        Preprocess the data by tokenizing text and organizing in specified format.
        
        Returns:
            List[dict]: List of processed data samples.
        """
        preprocessed = []
        for example in self.text:
            question = example['question']
            pos_document = example["context"]
            target = example['target']
            answer = example['answer']

            # Prepare question, context and instruction
            question_text = f"<QUESTION> {question} </QUESTION>"
            pos_document = f"<CONTEXT> {pos_document} </CONTEXT>"
            instruction_text = "<INST> Please identify and extract the <PA> sections that can answer the question (which may not be unique) </INST>"
            
            # Tokenize question and instruction separately
            question_tokens = self.tokenizer(
                question_text,
                truncation=False,
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids.squeeze()
            
            instruction_tokens = self.tokenizer(
                instruction_text,
                truncation=False,
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids.squeeze()
            
            # Calculate available length for context
            max_context_length = max(0, self.max_context_length - len(question_tokens) - len(instruction_tokens))
            if max_context_length == 0:
                continue
            
            # Truncate and tokenize context
            context_tokens = self.tokenizer(
                pos_document,
                truncation=True,
                max_length=max_context_length,
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids.squeeze()
            
            # Concatenate question, context and instruction tokens
            text_tokens = torch.full((self.max_context_length,), self.eos_tok_id, dtype=torch.long)
            concatenated_tokens = torch.cat((question_tokens, context_tokens, instruction_tokens), dim=0)
            text_tokens[:len(concatenated_tokens)] = concatenated_tokens
            
            # Tokenize answer
            a_tokens = self.tokenizer(answer, 
                                    return_tensors="pt",
                                    add_special_tokens=False).input_ids.squeeze()
            
            if a_tokens.dim() == 0:  # Handle single token answers
                a_tokens = a_tokens.unsqueeze(0)
            
            # Handle cases where length exceeds limit
            total_length = len(question_tokens) + len(a_tokens)
            if total_length > self.max_qa_len:
                available_length = self.max_qa_len
                if len(question_tokens) > available_length:
                    question_tokens = question_tokens[:available_length]
                    a_tokens = torch.tensor([], dtype=torch.long)  # No space left for answer tokens
                else:
                    remaining_length = available_length - len(question_tokens)
                    a_tokens = a_tokens[:remaining_length]
            
            # Prepare input IDs
            input_ids = torch.full((self.max_context_length + self.max_qa_len,), self.eos_tok_id, dtype=torch.long)
            input_ids[:self.max_context_length] = text_tokens
            input_ids[self.max_context_length:self.max_context_length+len(question_tokens)+len(a_tokens)] = torch.cat((question_tokens, a_tokens), dim=0)  

            # Target tokens: answer to the question
            target_tokens = torch.full((self.max_qa_len,), -100, dtype=torch.long)
            target_tokens[len(question_tokens)-1:len(question_tokens)-1+len(a_tokens)+1] = torch.cat((a_tokens, torch.tensor([self.eos_tok_id])), dim=0)

            preprocessed.append({
                "input_ids": input_ids,
                "labels": target_tokens,
                "query_length": len(question_tokens)
            })
            
        return preprocessed

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.preprocessed_data)

    def __getitem__(self, idx):
        """Get a single data sample."""
        sample = self.preprocessed_data[idx]
        return {
            "input_ids": sample["input_ids"],
            "labels": sample["labels"],
            "query_length": sample["query_length"]
        }

    def print_samples(self, num_samples=5):
        """Print sample data for inspection."""
        for i in range(min(num_samples, len(self.preprocessed_data))):
            sample = self.preprocessed_data[i]
            input_text = self.tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
            target_text = self.tokenizer.decode(sample["labels"], skip_special_tokens=True)
            print(f"Sample {i + 1}:\nInput: {input_text}\nTarget: {target_text}\n")
