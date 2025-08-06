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
    def __init__(self, text_file, llama_path, max_length, max_output_length, limit_size=None):
        """
        Initialize the dataset for text processing.

        Args:
            text_file (str): Path to the input text file (JSONL format).
            llama_path (str): Path to the pre-trained model.
            max_length (int): Maximum length of the input sequence.
            max_output_length (int): Maximum length of the output sequence.
            limit_size (int, optional): Limit the number of samples in the dataset (default is None).
        """
        self.text = read_jsonl_file(text_file)
        self.tokenizer = AutoTokenizer.from_pretrained(llama_path, use_auth_token="<to be filled>")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_context_length = max_length
        self.num_mem = 8  # Number of memory tokens, can be modified
        self.eos_tok_id = self.tokenizer.eos_token_id
        self.max_output_length = max_output_length
        self.preprocessed_data = self._preprocess_data()

        # Limit dataset size if specified
        if limit_size:
            self.preprocessed_data = self.preprocessed_data[:limit_size]
        print(f"Total dataset size: {len(self.preprocessed_data)}")

    def _preprocess_data(self):
        """
        Preprocess the data by tokenizing and structuring input-output pairs.

        Returns:
            List[Tuple[Tensor, Tensor, str, int]]: A list of tuples where each tuple contains 
            tokenized input, tokenized output, the target string, and the query length.
        """
        preprocessed = []
        for example in self.text:
            question = example['question']
            pos_document = example["context"]
            target = example['target']

            # Prepare question, context, and instruction as input components
            question_text = f"<QUESTION> {question} </QUESTION>"
            pos_document = f"<CONTEXT> {pos_document} </CONTEXT>"
            instruction_text = "<INST> Please identify and extract the <PA> sections that can answer the question (which may not be unique) </INST>"

            # Tokenize question and instruction separately
            question_tokens = self.tokenizer(question_text, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids.squeeze()
            instruction_tokens = self.tokenizer(instruction_text, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids.squeeze()

            # Calculate available length for context based on remaining tokens
            max_context_length = max(0, self.max_context_length - len(question_tokens) - len(instruction_tokens))

            # Tokenize context with truncation
            context_tokens = self.tokenizer(pos_document, truncation=True, padding="max_length", max_length=max_context_length, return_tensors="pt", add_special_tokens=False).input_ids.squeeze()

            # Concatenate question, context, and instruction tokens
            text_tokens = torch.cat((question_tokens, context_tokens, instruction_tokens), dim=0)

            # Tokenize the target (output sequence)
            gold_truth_tokens = self.tokenizer(target, truncation=True, padding="max_length", max_length=self.max_output_length, return_tensors="pt", add_special_tokens=False).input_ids.squeeze()

            # Append preprocessed data for the current example
            preprocessed.append((text_tokens, gold_truth_tokens, target, len(question_tokens)))

        return preprocessed

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.preprocessed_data)

    def __getitem__(self, idx):
        """
        Get a single example from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing tokenized input, target, and label information.
        """
        text_tokens, gold_truth_tokens, target, query_length = self.preprocessed_data[idx]

        # Prepare target tokens with padding (using -100 for tokens that should not be used in loss calculation)
        target_tokens = torch.full((len(gold_truth_tokens) + 1,), -100, dtype=torch.long)
        target_tokens[:len(gold_truth_tokens)] = gold_truth_tokens
        target_tokens[-1] = self.eos_tok_id  # EOS token for completion

        return {"input_ids": text_tokens, "ground_truths": gold_truth_tokens, "labels": target_tokens, "query_length": query_length}

    def print_samples(self, num_samples=5):
        """Print a few sample input-output pairs from the dataset."""
        for i in range(min(num_samples, len(self.preprocessed_data))):
            text_tokens, gold_truth_tokens, target, _ = self.preprocessed_data[i]
            input_text = self.tokenizer.decode(text_tokens, skip_special_tokens=True)
            target_text = self.tokenizer.decode(gold_truth_tokens, skip_special_tokens=True)
            print(f"Sample {i + 1}:\nInput: {input_text}\nTarget: {target_text}\n")
