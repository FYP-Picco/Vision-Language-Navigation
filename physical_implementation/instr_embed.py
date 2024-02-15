from transformers import BertTokenizer, AutoModel, BertModel

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained('bert-base-cased')

custom_text = "Walk past the striped area rug. Make a right onto the marble floor. Walk past the bathroom on the left. Make a left opposite the zebra painting. Walk through the open door, and wait at the mirror."

encoded_input = tokenizer.encode(custom_text, return_tensors='pt')

output = model(encoded_input)

print(output[0].size())