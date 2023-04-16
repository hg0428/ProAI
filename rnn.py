import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pickle import dump, load

# Define the transformer RNN model
class TransformerRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads):
        super(TransformerRNN, self).__init__()
        self.embed = nn.Embedding(input_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, num_heads), num_layers
        )
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = self.embed(x)
        x = x.transpose(0, 1)  # Swap batch and sequence length dimensions
        x = self.transformer(x)
        x = x.transpose(0, 1)  # Swap back to original dimensions
        x = self.linear(x)
        return x.reshape(
            -1, x.shape[-1]
        )  # Flatten batch and sequence length dimensions


# Define a dataset and dataloader for training
class TextCompletionDataset(Dataset):
    def __init__(self, data_dict, max_seq_len=20):
        self.data = []
        self.max_seq_len = max_seq_len
        for prefix, suffix in data_dict.items():
            prefix = [ord(c) for c in prefix]
            suffix = [ord(c) for c in suffix]
            if len(prefix) > max_seq_len:
                prefix = prefix[:max_seq_len]
                suffix = suffix[:max_seq_len]
            else:
                prefix = prefix + [0] * (max_seq_len - len(prefix))
                suffix = suffix + [0] * (max_seq_len - len(suffix))
            self.data.append(
                (
                    torch.tensor(prefix, dtype=torch.long),
                    torch.tensor(suffix, dtype=torch.long),
                )
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, batch):
        # Pad the last batch to the maximum sequence length
        max_seq_len = max(len(x[0]) for x in batch)
        padded_batch = []
        for prefix, suffix in batch:
            padded_prefix = torch.cat(
                [prefix, torch.zeros(max_seq_len - len(prefix), dtype=torch.long)]
            )
            padded_suffix = torch.cat(
                [suffix, torch.zeros(max_seq_len - len(suffix), dtype=torch.long)]
            )
            padded_batch.append((padded_prefix, padded_suffix))
        return padded_batch


batch_size = 4
train_dict = {
    "Hello": " World!",
    "Hello ": "World!",
    "hello": " world!",
    "Hello wor": "ld!",
    "helLo Wo": "rld!",
}
max_seq_len = max(
    max(len(x) for x in train_dict), max(len(train_dict[x]) for x in train_dict)
)

train_dataset = TextCompletionDataset(train_dict, max_seq_len=max_seq_len)
train_dataloader = DataLoader(
    train_dataset, batch_size=1, shuffle=True, collate_fn=train_dataset.collate_fn
)


train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)


# Train the model
input_size = 128  # ASCII character set
hidden_size = 256
output_size = 128  # ASCII character set
learning_rate = 0.25
num_epochs = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
try:
    with open("rnn", "rb") as f:
        model = load(f)
except:
    model = TransformerRNN(input_size, hidden_size, output_size, 8).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def save():
    with open("rnn", "wb") as f:
        dump(model, f)


print(f"Starting training on {device}.")

for epoch in range(num_epochs):
    for i, (prefix_tensor, suffix_tensor) in enumerate(train_dataloader):
        prefix_tensor = prefix_tensor.to(device)
        suffix_tensor = suffix_tensor.to(device)
        print("Here")
        optimizer.zero_grad()

        output = model(prefix_tensor)
        loss = criterion(output, suffix_tensor.view(-1))

        loss.backward()
        optimizer.step()

        if i % 1 == 0:
            print("Saving...")
            save()
            print("Saved!")
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}"
            )

# Test the model
def complete_text(model, prefix):
    with torch.no_grad():
        input_seq = (
            torch.tensor([ord(c) for c in prefix], dtype=torch.long)
            .unsqueeze(0)
            .to(device)
        )
        output_seq = prefix
        while len(output_seq) < 20:
            output = model(input_seq)
            output_idx = output.argmax(dim=1)[-1].item()
            output_seq += chr(output_idx)
            input_seq = torch.cat(
                [
                    input_seq[:, 1:],
                    torch.tensor([[output_idx]], dtype=torch.long).to(device),
                ],
                dim=1,
            )
        return output_seq


print("completion", complete_text(model, "Hello"))
