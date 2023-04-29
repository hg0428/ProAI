import string
from typing import Union
from json import load, dump

def tokenizer(x):
    return [tokens.index(i) for i in x]


def decoder(x):
    return [tokens[i] for i in x]


tokens = (
    [""]
    + [chr(x) for x in range(32, 65)]
    + [chr(x) for x in range(91, 127)]
    + list("\n")
)
Vocabularies = {
    'ascii': list('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n') + ['']
}

class Tokenizer:
    def __init__(self, vocabulary: list[Union[str, int, float]], binary:bool=False):
        print(vocabulary, len(vocabulary))
        self.vocabulary = vocabulary
        self.binary = binary
    def token_to_index(self, tok: Union[str, int, float]) -> int:
        return self.vocabulary.index(tok)
    def index_to_token(self, index: int) -> Union[str, int, float]:
        return self.vocabulary[tok]
    def tokenize(self, x: list[Union[str, int, float]]) -> list[int]:
        return [self.vocabulary.index(i) for i in x]
    def decode(self, x: list[int]) -> list[Union[str, int, float]]:
        return [self.vocabulary[i] for i in x]
    def from_save_file(filename: str) -> None:
        if type(filename) == Tokenizer:
            raise ValueError('`Tokenizer.from_save_file()` is a class method!')
        with open(filename, 'r') as f:
            data = load(f)
        return Tokenizer(data['vocabulary'], data['use_binary'])
    def save_to_file(self, filename: str) -> None:
        with open(filename, 'w') as f:
            dump({
                'vocabulary': self.vocabulary,
                'use_binary': self.binary
            }, f)
    def __call__(self, x):
        return self.tokenize(x)