import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset
from src.models.adapters import BottleneckAdapter

def freeze_base_model(model):
    for p in model.parameters():
        p.requires_grad = False

def insert_adapters(model, adapter_class):
    for layer in model.wav2vec2.encoder.layers:
        layer.adapter = adapter_class()
    return model

def train():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base")

    freeze_base_model(model)
    model = insert_adapters(model, BottleneckAdapter)

    train_data = load_dataset("audiofolder", data_dir="data/train")
    
    def preprocess(batch):
        batch["input_values"] = processor(batch["audio"]["array"], sampling_rate=16000).input_values[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    train_data = train_data.map(preprocess)

    loader = DataLoader(train_data["train"], batch_size=4, shuffle=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )

    model.train()
    for epoch in range(5):
        for batch in loader:
            optimizer.zero_grad()
            out = model(input_values=torch.tensor(batch["input_values"]).unsqueeze(0),
                        labels=torch.tensor(batch["labels"]).unsqueeze(0))
            loss = out.loss
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "weights/adapters/adapter_weights.pth")

if __name__ == "__main__":
    train()
