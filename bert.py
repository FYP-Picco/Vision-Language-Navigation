from transformers import BertTokenizer, BertModel
import torch

class BertImp:
    def __init__(self):
        # Load BERT tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.model = BertModel.from_pretrained('bert-base-cased')

    def encode_text(self,custom_text):
        tokens = self.tokenizer.encode(custom_text, return_tensors='pt')
        num_zeros_to_add = 200 - tokens.shape[1]
        zeros_tensor = torch.zeros((1, num_zeros_to_add),dtype=torch.int)
        result_tensor = torch.cat((tokens, zeros_tensor), dim=1)
        return result_tensor.squeeze().tolist()

    def get_output(self,tokens):
        return self.model(tokens)
