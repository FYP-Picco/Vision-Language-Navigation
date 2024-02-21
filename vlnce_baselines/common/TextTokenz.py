import nltk
#nltk.download('punkt')

import torch
from nltk.tokenize import word_tokenize

class TextProcessor:
    def __init__(self, vocab_file, device):
        with open(vocab_file) as f:
            self.vocab = f.read().split('", "')
        self.device = device

    def tokenize_text(self, text):
        tokens = word_tokenize(text.lower())
        return [self.vocab.index(token) for token in tokens if token in self.vocab] + [0] * (200 - len(tokens))

    def process(self, text):
        tokens = self.tokenize_text(text)
        token_tensor = torch.tensor([tokens], device=self.device)
        # batch = {'rgb': [], 'depth': [], 'instruction': token_tensor}
        batch = {'instruction': token_tensor}
        return batch

# Usage:
# processor = TextProcessor('data/Vocab_file.txt', torch.device('cuda:0'))
# text = "Follow the hallway until you see a gold colored trash can. Wait in front of the middle elevator."
# batch = processor.process(text)
# print(batch)