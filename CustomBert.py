import torch
from transformers import BertTokenizer, BertForMaskedLM
import torch.nn as nn
import torch.optim as optim
import random
import json
import pandas as pd
import tqdm

data = json.load(open('/home/tes/habitat-lab/VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/test/test.json'))

df = pd.json_normalize(data['episodes'])
df_instr = df['instruction.instruction_text']

# Define custom BERT model class
class CustomBertModel(nn.Module):
    def __init__(self, bert_model, additional_layer):
        super().__init__()
        self.bert = bert_model
        self.additional_layer = additional_layer

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0] # Use the [CLS] token representation
        embeddings = self.additional_layer(pooled_output)
        return embeddings


# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-cased')
additional_layer = nn.Linear(bert_model.config.hidden_size, 256)

# Resize additional layer's weight matrix
additional_layer.weight = nn.Parameter(additional_layer.weight[:, :512])

# Create custom model instance
custom_model = CustomBertModel(bert_model, additional_layer)

# Define optimizer
optimizer = optim.Adam(custom_model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    total_loss = 0
    for input_text in df_instr:
        # Tokenize input text and create MLM inputs
        input_ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors='pt')
        input_ids = input_ids.squeeze(0)  # Remove batch dimension
        max_length = 512
        if len(input_ids) < max_length:
            input_ids = torch.cat([input_ids, torch.tensor([tokenizer.pad_token_id] * (max_length - len(input_ids)))], dim=-1)
        else:
            input_ids = input_ids[:max_length]
        input_ids = input_ids.unsqueeze(0)  # Add batch dimension
        attention_mask = (input_ids != tokenizer.pad_token_id).float()
        labels = input_ids.clone()  # Labels are the same as input_ids for MLM
        masked_indices = torch.rand(input_ids.shape) < 0.15  # Mask 15% of tokens
        masked_indices[input_ids == tokenizer.pad_token_id] = False  # Exclude padding tokens from masking
        input_ids[masked_indices] = tokenizer.mask_token_id

        # Forward pass
        outputs = custom_model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = outputs
        loss = criterion(predictions.view(-1, tokenizer.vocab_size), labels.view(-1))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Print training progress
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_dataset):.4f}')

# Save model checkpoint to .pth file
torch.save(custom_model.state_dict(), 'custom_bert_model_mlm.pth')

# Load model from .pth file
loaded_model = CustomBertModel(bert_model, additional_layer)
loaded_model.load_state_dict(torch.load('custom_bert_model_mlm.pth'))
loaded_model.eval()  # Set model to evaluation mode

