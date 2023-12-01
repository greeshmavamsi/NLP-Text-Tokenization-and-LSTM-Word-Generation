
# Install necessary libraries
!pip install transformers
import nltk
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Step 1: Download necessary NLTK data and spaCy model
nltk.download('stopwords')
nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')

# Step 2: Tokenize using Transformers package
def tokenize_with_transformers(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokens = tokenizer.tokenize(text)
    return tokens

# Step 3: Tokenize with stopwords as delimiters using NLTK
def tokenize_with_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    tokens = [word.lower() for word in words if word.lower() not in stop_words]
    return tokens

# Step 4: Tokenize using spaCy
def tokenize_with_spacy(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop]
    return tokens

# Step 5: Create a custom Dataset
class WordDataset(Dataset):
    def __init__(self, texts, tokenizer, transform=None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if self.transform:
            text = self.transform(text)
        tokens = self.tokenizer(text)
        return tokens

# Step 6: Define LSTMWordGenerator model
class LSTMWordGenerator(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(LSTMWordGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out)
        return output

# Step 7: Preprocess and tokenize your text data
text_data = "This is a sample text. Replace this with your actual text data."
transformed_text = tokenize_with_transformers(text_data)
stopwords_text = tokenize_with_stopwords(text_data)
spacy_text = tokenize_with_spacy(text_data)

# Step 8: Create Dataloader using WordDataset
transformer_dataset = WordDataset(transformed_text, tokenizer=tokenize_with_transformers)
dataloader = DataLoader(transformer_dataset, batch_size=16, shuffle=True)

# Step 9: Training loop
num_epochs = 10
vocab_size = 10000  # Replace with your actual vocabulary size
embed_size = 128
hidden_size = 256
lr = 0.001

model = LSTMWordGenerator(vocab_size, embed_size, hidden_size)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_tuple in dataloader:
        batch = []
        max_length = max(len(item) for item in batch_tuple)

        for item in batch_tuple:
            if isinstance(item, int):
                tensor = torch.tensor([item])
            elif isinstance(item, str):
                tensor = torch.zeros(max_length, dtype=torch.long)
                if len(item) > 0:
                    tensor[:len(item)] = torch.tensor(item.encode())
            batch.append(tensor)

        batch = torch.stack(batch, dim=0)

        optimizer.zero_grad()
        outputs = model(batch)  # Pass the batch of tokenized data
        targets = batch  # Targets are the same as the input sequences
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}')
