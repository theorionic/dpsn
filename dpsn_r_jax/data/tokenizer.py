import numpy as np
from typing import List, Union, Any
from transformers import AutoTokenizer


def get_tokenizer(name_or_path: str = None, max_val: int = 100) -> Any:
    if name_or_path and name_or_path.lower() != "numeric":
        tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    return SimpleNumberTokenizer(max_val=max_val)


class SimpleNumberTokenizer:
    def __init__(self, max_val=100):
        self.max_val = max_val
        self.pad_token = "<PAD>"
        self.eos_token = "<EOS>"
        self.sep_token = "->"
        self.unk_token = "<UNK>"

        self.pad_token_id = max_val
        self.eos_token_id = max_val + 1
        self.sep_token_id = max_val + 2
        self.unk_token_id = max_val + 3

        self.vocab_size = max_val + 4

        self.special_tokens = {
            self.pad_token: self.pad_token_id,
            self.eos_token: self.eos_token_id,
            self.sep_token: self.sep_token_id,
            self.unk_token: self.unk_token_id,
        }
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}

    def encode(
        self, text: str, return_tensors: str = None
    ) -> Union[List[int], np.ndarray]:
        # Simple splitting by space
        tokens = text.split()
        ids = []
        for token in tokens:
            if token in self.special_tokens:
                ids.append(self.special_tokens[token])
            elif token.isdigit() and int(token) < self.max_val:
                ids.append(int(token))
            else:
                ids.append(self.unk_token_id)

        if return_tensors == "np":
            return np.array([ids])
        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for tid in token_ids:
            if tid in self.id_to_token:
                if not skip_special_tokens:
                    tokens.append(self.id_to_token[tid])
            elif tid < self.max_val:
                tokens.append(str(tid))
            else:
                if not skip_special_tokens:
                    tokens.append(self.unk_token)
        return " ".join(tokens)

    def __len__(self):
        return self.vocab_size
